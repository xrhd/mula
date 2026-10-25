"""Plots the reward history of a run (eval mean, eval max and record) as a PNG.

Usage:
  uv run --package ppo python plot.py runs/minatar-breakout-seed0   # → reward.png
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

import render
from envs import ENVS

W, H = 900, 420
BOX = (80, 60, W - 30, H - 60)  # plot area
MAX_COLOR = (90, 90, 110)


def plot(run: Path) -> Path:
  metrics = [json.loads(line) for line in (run / "metrics.jsonl").open()]
  cfg = json.loads((run / "config.json").read_text())
  epochs = [m["epoch"] for m in metrics]
  series = {
      "eval máx": ([m["eval_max"] for m in metrics], MAX_COLOR, 2, False),
      "eval médio": ([m["eval_mean"] for m in metrics], render.CURVE, 3, False),
      "recorde": ([m["record"] for m in metrics], render.ACCENT, 2, True),
  }
  x0, y0, x1, y1 = BOX
  max_e = max(epochs) or 1
  max_y = max(max(v) for v, *_ in series.values()) or 1.0
  to_xy = lambda e, y: (x0 + (x1 - x0) * e / max_e, y1 - (y1 - y0) * y / max_y)

  img = Image.new("RGB", (W, H), render.BG)
  d = ImageDraw.Draw(img)
  f = render.font(14)
  d.text((x0, 18), f"{ENVS[cfg['env']].title} · PPO · histórico de reward",
         fill=render.FG, font=render.font(22))
  for i in range(5):  # horizontal grid + y labels
    y = max_y * i / 4
    _, py = to_xy(0, y)
    d.line((x0, py, x1, py), fill=(40, 40, 50))
    d.text((x0 - 8, py), render.fmt_int(y), fill=render.MUTED, font=f, anchor="rm")
  for e in range(0, max_e + 1, max(1, max_e // 6)):
    px, _ = to_xy(e, 0)
    d.text((px, y1 + 8), str(e), fill=render.MUTED, font=f, anchor="ma")
  d.text(((x0 + x1) / 2, y1 + 30), "época", fill=render.MUTED, font=f, anchor="ma")

  for values, color, width, step in series.values():
    pts = [to_xy(e, v) for e, v in zip(epochs, values)]
    if step:  # the record only changes at record epochs
      pts = [q for a, b in zip(pts, pts[1:]) for q in (a, (b[0], a[1]))] + pts[-1:]
    d.line(pts, fill=color, width=width)
  for m in metrics:
    if m["new_record"]:
      cx, cy = to_xy(m["epoch"], m["record"])
      d.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=render.ACCENT)

  lx = x0 + 12
  for name, (_, color, *_) in series.items():  # legend
    d.line((lx, y0 + 10, lx + 20, y0 + 10), fill=color, width=3)
    d.text((lx + 26, y0 + 10), name, fill=render.FG, font=f, anchor="lm")
    lx += 40 + d.textlength(name, font=f)

  out = run / "reward.png"
  img.save(out)
  return out


if __name__ == "__main__":
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("run_dirs", type=Path, nargs="+")
  for run in p.parse_args().run_dirs:
    print(plot(run))
