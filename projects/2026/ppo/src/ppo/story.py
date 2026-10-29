"""Builds the "training a neural network from scratch" video of a run.

For every record (in order): a title card, then the *whole* record episode,
re-simulated from its replay checkpoint. Ends with the full learning curve.
Frames are streamed to ffmpeg (bundled by imageio-ffmpeg).

Usage:
  uv run ppo-story runs/minatar-breakout-seed0
  uv run ppo-story runs/2048-seed0 --fps 60 --max-clip-seconds 40
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from PIL import ImageDraw

from ppo import render
from ppo.envs import ENVS


class Mp4Writer:

  def __init__(self, path: Path, fps: float):
    self.count = 0
    self.proc = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{render.W}x{render.H}",
            "-r", str(fps), "-i", "-", "-vcodec", "libx264", "-pix_fmt",
            "yuv420p", "-crf", "20", "-preset", "medium", str(path),
        ],
        stdin=subprocess.PIPE,
    )
    self.fps = fps

  def write(self, img, repeat: int = 1):
    data = img.tobytes()
    for _ in range(repeat):
      self.proc.stdin.write(data)
    self.count += repeat

  def hold(self, img, seconds: float):
    self.write(img, max(1, round(seconds * self.fps)))

  def close(self):
    self.proc.stdin.close()
    if self.proc.wait() != 0:
      sys.exit("ffmpeg failed")


def read_jsonl(path: Path):
  return [json.loads(line) for line in path.open()] if path.exists() else []


def parse_args():
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("run_dir", type=Path)
  p.add_argument("--out", type=Path, help="default: <run_dir>/story.mp4")
  p.add_argument("--fps", type=float, default=30.0)
  p.add_argument(
      "--max-clip-seconds", type=float, default=None,
      help="speed long episodes up (skipping frames) to fit; off by default",
  )
  p.add_argument("--card-seconds", type=float, default=1.5)
  p.add_argument("--platform", help="JAX platform (default: the training one)")
  return p.parse_args()


def main():
  args = parse_args()
  run = args.run_dir
  records = read_jsonl(run / "records.jsonl")
  metrics = read_jsonl(run / "metrics.jsonl")
  if not records:
    sys.exit(f"no records in {run}: train with main.py first")
  cfg = json.loads((run / "config.json").read_text())
  os.environ.setdefault("JAX_PLATFORMS", args.platform or cfg["platform"])
  spec = ENVS[cfg["env"]]
  record_epochs = [r["epoch"] for r in records]
  title = f"{spec.title} · PPO do zero"

  out = args.out or run / "story.mp4"
  video = Mp4Writer(out, args.fps)
  video.hold(render.card([
      ("Treinando uma IA do zero", 48, render.FG),
      (f"{spec.title} · PPO · flax.nnx · pgx", 26, render.MUTED),
      (f"{len(records)} recordes em {metrics[-1]['epoch']} épocas", 26, render.ACCENT),
  ]), 2.5)

  manifest = []
  for r in records:
    record = render.load_record(run / "records" / r["path"])
    states, total = render.replay(record)
    n_steps = len(states) - 1
    if abs(total - record["return"]) > 1e-3:
      print(
          f"warning: record {r['record']} replayed {total} != saved"
          f" {record['return']} (different JAX platform?)", file=sys.stderr,
      )
    stride = 1
    if args.max_clip_seconds:
      stride = max(1, math.ceil(len(states) / (args.max_clip_seconds * args.fps)))

    prev = r["previous"]
    video.hold(render.card([
        (f"Época #{r['epoch']}", 56, render.FG),
        ("NOVO RECORDE!", 40, render.ACCENT),
        (
            f"{render.fmt_int(r['return'])} pontos"
            + ("" if prev is None else f" (antes {render.fmt_int(prev)})"),
            28, render.FG,
        ),
        (f"{render.fmt_steps(r['steps'])} passos de treino", 22, render.MUTED),
    ]), args.card_seconds)

    start = video.count
    last = None
    for frame in render.clip_frames(
        record, states, spec.render_kind, title, metrics, record_epochs, stride
    ):
      video.write(frame)
      last = frame
    clip_frames = video.count - start
    video.hold(last, 0.7)
    manifest.append({
        "record": r["record"], "epoch": r["epoch"], "return": total,
        "episode_steps": n_steps, "stride": stride, "frames": clip_frames,
    })
    print(
        f"recorde {r['record']:3d} | época {r['epoch']:4d} |"
        f" {render.fmt_int(total):>7} pontos | {n_steps} passos |"
        f" {clip_frames} frames", flush=True,
    )

  final = render.card([
      ("Fim do treino", 44, render.FG),
      (
          f"recorde final: {render.fmt_int(records[-1]['return'])} pontos ·"
          f" média do eval: {metrics[-1]['eval_mean']:.1f}", 24, render.ACCENT,
      ),
  ])
  render.draw_curve(
      ImageDraw.Draw(final), (120, 360, render.W - 120, render.H - 40), metrics,
      None, record_epochs,
  )
  video.hold(final, 4.0)
  video.close()

  manifest_path = out.with_name(f"{out.stem}_manifest.json")
  manifest_path.write_text(json.dumps(manifest, indent=2))
  print(f"{out} ({video.count / args.fps:.0f}s, {video.count} frames)")


if __name__ == "__main__":
  main()
