"""Plots eval return vs observed steps, one figure per environment."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# pgx examples/minatar-ppo, PPO at 20M steps
BENCHMARKS = {"minatar-freeway": 63.5, "minatar-breakout": 50.0}
LABELS = {"ppo": "PPO (GAE)", "phi_ppo": "ϕ-PPO + PTR"}


def main(results_dir: Path):
  runs = defaultdict(list)
  for path in sorted(results_dir.glob("*.jsonl")):
    env_name, algo, _ = path.stem.rsplit("-", 2)
    runs[env_name].append((algo, [json.loads(line) for line in path.open()]))

  for env_name, algo_rows in runs.items():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo, rows in algo_rows:
      xs = [r["observed_steps"] / 1e6 for r in rows]
      ys = [r["eval_R"] for r in rows]
      ax.plot(xs, ys, label=f"{LABELS.get(algo, algo)} — final {ys[-1]:.1f}")
    if env_name in BENCHMARKS:
      b = BENCHMARKS[env_name]
      ax.axhline(b, ls="--", c="gray", lw=1, label=f"benchmark pgx ({b})")
    ax.set_xlabel("passos observados (milhões)")
    ax.set_ylabel("eval return")
    ax.set_title(f"{env_name} — Apple M5 (jax-mps)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = results_dir / f"{env_name}_eval_return.png"
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
  default = Path(__file__).parent / "results"
  main(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
