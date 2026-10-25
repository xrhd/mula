"""Replays the best evaluation episode saved by train.py and renders it.

train.py (save_best=true) stores the params, the evaluation key and the index
of the best episode. Evaluation is deterministic given those, so the episode
is re-simulated here and only its states are kept: no cost during training.

  * MinAtar: GIF (or MP4) drawn with the pgx-minatar colormap. pgx itself
    does not animate MinAtar (`save_svg_animation` asserts against it).
  * 2048: GIF (or MP4) plus pgx's own animated SVG (`pgx.save_svg_animation`).

Usage:
  uv run --package causal-credit-assignment python render.py
  uv run --package causal-credit-assignment python render.py \
      results/best/minatar-breakout-phi_ppo-seed0.pkl --format mp4 --fps 30
"""

import argparse
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).parent


def parse_args():
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("path", nargs="?", help="best-episode .pkl (default: newest)")
  p.add_argument("--out", help="output file (default: next to the .pkl)")
  p.add_argument("--format", choices=["gif", "mp4"], default="gif")
  p.add_argument("--fps", type=float, default=30.0)
  p.add_argument("--scale", type=int, default=32, help="pixels per cell")
  p.add_argument("--stride", type=int, default=1, help="keep every k-th frame")
  p.add_argument(
      "--platform", help="JAX platform (default: the one used for training)"
  )
  return p.parse_args()


def newest_best() -> Path:
  files = sorted((HERE / "results" / "best").glob("*.pkl"), key=os.path.getmtime)
  if not files:
    sys.exit("no results/best/*.pkl found: train with save_best=true first")
  return files[-1]


def replay(env, forward, params, eval_key, num_envs: int, episode: int):
  """Re-simulates the evaluation batch, keeping the states of `episode`."""
  import jax

  from agent import eval_init, eval_step

  step = jax.jit(lambda s, k: eval_step(env, forward, params, s, k))
  state, key = eval_init(env, jax.numpy.asarray(eval_key), num_envs)
  pick = lambda s: jax.tree_util.tree_map(lambda x: np.asarray(x[episode]), s)
  states, total = [pick(state)], 0.0
  while not bool(state.terminated[episode]):
    state, key = step(state, key)
    states.append(pick(state))
    total += float(state.rewards[episode, 0])
  return states, total


def minatar_frame(state, scale: int) -> np.ndarray:
  from pgx_minatar.utils import get_cmap

  obs = np.asarray(state.observation)
  n_channels = obs.shape[-1]
  cmap = np.array([(0, 0, 0)] + get_cmap(n_channels)) * 255
  # same layering as pgx_minatar.utils.visualize_minatar
  index = (obs * (np.arange(n_channels) + 1)).max(-1)
  img = cmap[index].astype(np.uint8)
  return np.kron(img, np.ones((scale, scale, 1), np.uint8))


TILE_COLORS = [
    (205, 193, 180), (238, 228, 218), (237, 224, 200), (242, 177, 121),
    (245, 149, 99), (246, 124, 95), (246, 94, 59), (237, 207, 114),
    (237, 204, 97), (237, 200, 80), (237, 197, 63), (237, 194, 46),
]


def play2048_frame(state, scale: int) -> np.ndarray:
  board = np.asarray(state._board).reshape(4, 4)
  cell = 2 * scale
  img = Image.new("RGB", (4 * cell, 4 * cell), (187, 173, 160))
  draw = ImageDraw.Draw(img)
  for r in range(4):
    for c in range(4):
      b = int(board[r, c])
      box = (c * cell + 4, r * cell + 4, (c + 1) * cell - 4, (r + 1) * cell - 4)
      draw.rectangle(box, fill=TILE_COLORS[min(b, len(TILE_COLORS) - 1)])
      if b:
        draw.text(
            ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), str(2**b),
            fill=(50, 50, 50), anchor="mm",
        )
  return np.asarray(img)


def with_header(frame: np.ndarray, text: str) -> Image.Image:
  h, w, _ = frame.shape
  img = Image.new("RGB", (w, h + 18), (255, 255, 255))
  img.paste(Image.fromarray(frame), (0, 18))
  ImageDraw.Draw(img).text((4, 3), text, fill=(0, 0, 0))
  return img


def write_gif(frames, path: Path, fps: float):
  frames[0].save(
      path, save_all=True, append_images=frames[1:],
      duration=max(20, int(1000 / fps)), loop=0, optimize=True,
  )


def write_mp4(frames, path: Path, fps: float):
  if shutil.which("ffmpeg") is None:
    sys.exit("--format mp4 needs ffmpeg (brew install ffmpeg); use --format gif")
  w, h = frames[0].size
  w, h = w - w % 2, h - h % 2  # yuv420p needs even sizes
  cmd = [
      "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt",
      "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-pix_fmt",
      "yuv420p", "-vcodec", "libx264", str(path),
  ]
  proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
  for f in frames:
    proc.stdin.write(np.asarray(f.crop((0, 0, w, h))).tobytes())
  proc.stdin.close()
  proc.wait()


def main():
  args = parse_args()
  path = Path(args.path) if args.path else newest_best()
  with open(path, "rb") as f:
    best = pickle.load(f)
  cfg = best["config"]
  os.environ.setdefault("JAX_PLATFORMS", args.platform or cfg["platform"])

  import pgx
  from flax import nnx

  from agent import make_forward

  env = pgx.make(cfg["env_name"])
  params, forward = make_forward(env, cfg["seed"])
  nnx.replace_by_pure_dict(params, best["params"])

  states, total = replay(
      env, forward, params, best["eval_key"], cfg["num_eval_envs"],
      best["episode"],
  )
  print(
      f"{path.name}: iter {best['iter']}, {best['observed_steps']:,} observed"
      f" steps, saved return {best['return']:.1f}, replayed {total:.1f}"
      f" ({len(states) - 1} steps)"
  )
  if abs(total - best["return"]) > 1e-3:
    print(
        "warning: replay differs from the saved episode (different JAX"
        " platform/version?); rendering the replayed one",
        file=sys.stderr,
    )

  is_minatar = cfg["env_name"].startswith("minatar")
  draw = minatar_frame if is_minatar else play2048_frame
  frames, score = [], 0.0
  for t, s in enumerate(states):
    score += float(s.rewards[0]) if t else 0.0
    if t % args.stride == 0 or t == len(states) - 1:
      frames.append(with_header(draw(s, args.scale), f"t={t}  return={score:g}"))

  out = Path(args.out) if args.out else path.with_suffix(f".{args.format}")
  (write_gif if args.format == "gif" else write_mp4)(frames, out, args.fps)
  print(out)
  if not is_minatar:  # pgx's recommended animation for board games
    svg = out.with_suffix(".svg")
    pgx.save_svg_animation(states[:: args.stride], svg, frame_duration_seconds=0.2)
    print(svg)


if __name__ == "__main__":
  main()
