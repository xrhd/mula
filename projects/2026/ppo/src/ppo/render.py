"""Re-simulates record episodes and draws video frames (960x540).

Layout: the game on the left, a panel on the right with the epoch, training
steps, the record, the live score and the learning curve up to that epoch.
"""

import math
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 960, 540
GAME = 480  # game area: GAME x GAME, vertically centered on the left
PANEL_X = 520
BG = (18, 18, 24)
FG = (235, 235, 240)
MUTED = (140, 140, 155)
ACCENT = (255, 196, 0)
CURVE = (90, 170, 255)


# Pillow's default font has no accents ("Época"): prefer a system TTF.
FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",  # macOS
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    "C:/Windows/Fonts/arial.ttf",
]
_fonts = {}


def font(size: int):
  if size not in _fonts:
    for path in FONT_CANDIDATES:
      if Path(path).exists():
        _fonts[size] = ImageFont.truetype(path, size)
        break
    else:
      _fonts[size] = ImageFont.load_default(size=size)
  return _fonts[size]


def fmt_int(x: float) -> str:
  return f"{x:,.0f}".replace(",", ".")


def fmt_steps(steps: int) -> str:
  if steps >= 1e6:
    return f"{steps / 1e6:.2f}M".replace(".", ",")
  return f"{steps / 1e3:.0f}k"


# ---------------------------------------------------------------- replay


def load_record(path: Path) -> dict:
  with open(path, "rb") as f:
    return pickle.load(f)


def replay(record: dict):
  """Re-simulates the recorded evaluation and returns (states, return).

  Must run on the same JAX platform as training to be bit-exact.
  """
  import jax
  import pgx
  from flax import nnx

  from ppo.agent import eval_init, eval_step, make_forward
  from ppo.envs import ENVS

  cfg = record["config"]
  env = pgx.make(ENVS[cfg["env"]].pgx_id)
  params, forward = make_forward(env, cfg["seed"])
  nnx.replace_by_pure_dict(params, record["params"])

  step = jax.jit(lambda s, k: eval_step(env, forward, params, s, k))
  state, key = eval_init(env, jax.numpy.asarray(record["eval_key"]), cfg["num_eval_envs"])
  i = record["episode"]
  pick = lambda s: jax.tree_util.tree_map(lambda x: np.asarray(x[i]), s)
  states, total = [pick(state)], 0.0
  while not bool(state.terminated[i]):
    state, key = step(state, key)
    states.append(pick(state))
    total += float(state.rewards[i, 0])
  return states, total


# ---------------------------------------------------------------- games


def minatar_frame(state) -> Image.Image:
  from pgx_minatar.utils import get_cmap

  obs = np.asarray(state.observation)
  n_channels = obs.shape[-1]
  cmap = np.array([(0, 0, 0)] + get_cmap(n_channels)) * 255
  index = (obs * (np.arange(n_channels) + 1)).max(-1)  # as visualize_minatar
  img = Image.fromarray(cmap[index].astype(np.uint8))
  return img.resize((GAME, GAME), Image.NEAREST)


TILE_COLORS = [
    (205, 193, 180), (238, 228, 218), (237, 224, 200), (242, 177, 121),
    (245, 149, 99), (246, 124, 95), (246, 94, 59), (237, 207, 114),
    (237, 204, 97), (237, 200, 80), (237, 197, 63), (237, 194, 46),
]


def play2048_frame(state) -> Image.Image:
  board = np.asarray(state._board).reshape(4, 4)
  cell = GAME // 4
  img = Image.new("RGB", (GAME, GAME), (187, 173, 160))
  draw = ImageDraw.Draw(img)
  f = font(36)
  for r in range(4):
    for c in range(4):
      b = int(board[r, c])
      box = (c * cell + 6, r * cell + 6, (c + 1) * cell - 6, (r + 1) * cell - 6)
      draw.rounded_rectangle(
          box, radius=8, fill=TILE_COLORS[min(b, len(TILE_COLORS) - 1)]
      )
      if b:
        draw.text(
            ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), str(2**b),
            fill=(60, 50, 40) if b < 3 else (255, 255, 255), font=f,
            anchor="mm",
        )
  return img


GAME_FRAMES = {"minatar": minatar_frame, "2048": play2048_frame}


# ---------------------------------------------------------------- panel


def draw_curve(draw, box, metrics, upto_epoch=None, records=()):
  """Eval mean vs epoch, up to `upto_epoch`, with record epochs marked."""
  x0, y0, x1, y1 = box
  draw.rectangle(box, outline=(60, 60, 75))
  pts = [(m["epoch"], m["eval_mean"]) for m in metrics]
  if not pts:
    return
  max_epoch = max(e for e, _ in pts) or 1
  max_y = max(max(y for _, y in pts), 1e-6)
  shown = [(e, y) for e, y in pts if upto_epoch is None or e <= upto_epoch]
  to_xy = lambda e, y: (
      x0 + 8 + (x1 - x0 - 16) * e / max_epoch,
      y1 - 8 - (y1 - y0 - 16) * y / max_y,
  )
  if len(shown) > 1:
    draw.line([to_xy(e, y) for e, y in shown], fill=CURVE, width=3)
  for e in records:
    if upto_epoch is None or e <= upto_epoch:
      ys = [y for ee, y in pts if ee == e]
      if ys:
        cx, cy = to_xy(e, ys[0])
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=ACCENT)
  f = font(14)
  draw.text((x0 + 6, y0 + 4), f"média do eval (máx {max_y:.1f})", fill=MUTED, font=f)
  draw.text((x1 - 6, y1 + 4), f"época {max_epoch}", fill=MUTED, font=f, anchor="ra")


def clip_background(title, record, metrics, record_epochs) -> Image.Image:
  """Static part of every frame of a record clip."""
  img = Image.new("RGB", (W, H), BG)
  d = ImageDraw.Draw(img)
  d.text((PANEL_X, 34), title, fill=MUTED, font=font(22))
  d.text((PANEL_X, 64), f"Época #{record['epoch']}", fill=FG, font=font(52))
  d.text(
      (PANEL_X, 130), f"{fmt_steps(record['steps'])} passos de treino",
      fill=MUTED, font=font(22),
  )
  prev = record["previous"]
  sub = "primeiro jogo" if prev is None else f"antes: {fmt_int(prev)}"
  d.text((PANEL_X, 172), f"Recorde #{record['record']}", fill=ACCENT, font=font(26))
  d.text(
      (PANEL_X, 204), f"{fmt_int(record['return'])} pontos ({sub})",
      fill=ACCENT, font=font(22),
  )
  draw_curve(
      d, (PANEL_X, 330, W - 30, H - 40), metrics, record["epoch"], record_epochs
  )
  return img


def compose(background: Image.Image, game: Image.Image, score, t, n_steps):
  img = background.copy()
  img.paste(game, (20, (H - GAME) // 2))
  d = ImageDraw.Draw(img)
  d.text((PANEL_X, 250), f"pontos: {fmt_int(score)}", fill=FG, font=font(30))
  d.text((PANEL_X, 290), f"passo {t}/{n_steps}", fill=MUTED, font=font(18))
  return img


def card(lines) -> Image.Image:
  """Full-screen title card; `lines` = [(text, size, color), ...]."""
  img = Image.new("RGB", (W, H), BG)
  d = ImageDraw.Draw(img)
  total = sum(int(s * 1.4) for _, s, _ in lines)
  y = (H - total) / 2
  for text, size, color in lines:
    d.text((W / 2, y), text, fill=color, font=font(size), anchor="ma")
    y += int(size * 1.4)
  return img


def clip_frames(record, states, render_kind, title, metrics, record_epochs,
                stride: int = 1):
  """Yields the frames of a whole record episode (every `stride` steps)."""
  bg = clip_background(title, record, metrics, record_epochs)
  draw_game = GAME_FRAMES[render_kind]
  n = len(states) - 1
  score = 0.0
  for t, s in enumerate(states):
    if t:
      score += float(s.rewards[0])
    if t % stride == 0 or t == n:
      yield compose(bg, draw_game(s), score, t, n)


def n_frames(n_states: int, stride: int) -> int:
  n = n_states - 1
  return math.ceil(n_states / stride) + (0 if n % stride == 0 else 1)
