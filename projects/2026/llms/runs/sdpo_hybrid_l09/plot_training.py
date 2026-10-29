"""Painel de treino do run SDPO híbrido (λ=0.9) a partir dos eventos do Tensorboard.

uvx --with tensorboard --with matplotlib --with numpy python plot_training.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SERIES, TEACHER = "#2a78d6", "#eb6834"  # slots 1 e 2 da paleta de referência
WARMUP_END = 336  # 10% de MAX_STEPS: pico do learning rate
SMOOTH = 25

events = sorted(glob.glob(os.path.join(HERE, "tb", "events*")), key=os.path.getsize)[-1]
ea = EventAccumulator(events, size_guidance={"scalars": 0})
ea.Reload()


def series(tag):
  ev = ea.Scalars(tag)
  return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def rolling(y, k=SMOOTH):
  if len(y) < k:
    return y
  c = np.cumsum(np.insert(y, 0, 0.0))
  out = (c[k:] - c[:-k]) / k
  return np.concatenate([np.full(k - 1, np.nan), out])


steps_act, act = series("actor/train/sdpo/active_token_frac")
teacher_steps = steps_act[act > 0]

panels = [
    ("Formato exato (treino, %)", "rewards/train/match_format_exactly", lambda v: v / 3 * 100, "linear"),
    ("Resposta certa: check_answer (treino)", "rewards/train/check_answer", None, "linear"),
    ("KL vs. referência", "actor/train/kl", None, "symlog"),
    ("Norma do gradiente (antes do clip)", "actor/train/grad_norm", None, "log"),
    ("Tokens com teacher (%)", "actor/train/sdpo/active_token_frac", lambda v: v * 100, "linear"),
    ("Comprimento da resposta (tokens)", "completions/train/mean_length", None, "linear"),
    ("Entropia", "actor/train/entropy", None, "linear"),
    ("Validação: check_answer", "rewards/eval/check_answer", None, "linear"),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
})
fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True, facecolor=SURFACE)
last_step = int(steps_act.max())
for ax, (title, tag, fn, scale) in zip(axes.flat, panels):
  ax.set_facecolor(SURFACE)
  x, y = series(tag)
  if fn:
    y = fn(y)
  if tag.startswith("rewards/eval"):
    ax.plot(x, y, color=SERIES, lw=2, marker="o", ms=4, mec=SURFACE, mew=1)
  else:
    ax.plot(x, y, color=SERIES, lw=0.6, alpha=0.25)
    ax.plot(x, rolling(y), color=SERIES, lw=2)
  ax.set_yscale(scale)
  ymin, ymax = ax.get_ylim()
  # Rug com os passos em que o teacher atuou, na base de cada painel.
  ax.vlines(teacher_steps, 0, 0.06, transform=ax.get_xaxis_transform(), color=TEACHER, lw=1)
  ax.axvline(WARMUP_END, color=MUTED, lw=1, ls=(0, (3, 3)))
  ax.set_title(title, loc="left")
  ax.grid(axis="y", color=GRID, lw=0.8)
  ax.spines[["top", "right"]].set_visible(False)
  ax.tick_params(length=0)
axes[0, 1].annotate("pico do LR", xy=(WARMUP_END, 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -10), textcoords="offset points", color=MUTED, fontsize=8)
for ax in axes[-1]:
  ax.set_xlabel("passo de treino")

handles = [
    plt.Line2D([], [], color=SERIES, lw=2, label=f"média móvel ({SMOOTH} passos)"),
    plt.Line2D([], [], color=SERIES, lw=0.6, alpha=0.4, label="valor por passo"),
    plt.Line2D([], [], color=TEACHER, lw=1, marker="|", ms=10, ls="", label="passo com teacher (SDPO ativo)"),
    plt.Line2D([], [], color=MUTED, lw=1, ls=(0, (3, 3)), label="fim do warmup / pico do LR"),
]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.955),
           labelcolor=INK_2)
fig.suptitle(f"SDPO híbrido (λ=0.9) — Gemma-3-1B-IT / GSM8K — até o passo {last_step} de 3364",
             x=0.06, y=0.985, ha="left", color=INK, fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = os.path.join(HERE, "training_panel.png")
fig.savefig(out, dpi=140, facecolor=SURFACE)
print(out, "| teacher steps:", len(teacher_steps), "| last step:", last_step)
