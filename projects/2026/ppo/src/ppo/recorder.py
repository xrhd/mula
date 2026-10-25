"""Keeps track of record-breaking evaluation episodes.

A record is stored as a *replay checkpoint* (params + evaluation key + episode
index) instead of frames: evaluation is deterministic, so render.py can
re-simulate the episode afterwards. Saving costs ~300 KB and only happens
when the record improves.
"""

import json
import pickle
import shutil
from pathlib import Path

import jax
import numpy as np
from flax import nnx


class RecordKeeper:

  def __init__(self, run_dir: Path, config: dict):
    self.dir = Path(run_dir) / "records"
    self.log_path = Path(run_dir) / "records.jsonl"
    shutil.rmtree(self.dir, ignore_errors=True)  # a new run starts from zero
    self.dir.mkdir(parents=True)
    self.log_path.unlink(missing_ok=True)
    self.config = config
    self.best = -np.inf
    self.count = 0

  def update(self, returns, params, eval_key, epoch: int, steps: int, sec: float):
    """Saves the best episode of this evaluation if it beats the record."""
    returns = np.asarray(returns)
    episode = int(returns.argmax())
    value = float(returns[episode])
    if value <= self.best:
      return False

    self.count += 1
    previous = None if np.isinf(self.best) else self.best
    self.best = value
    path = self.dir / f"{self.count:03d}_epoca-{epoch:04d}.pkl"
    info = {
        "record": self.count, "epoch": epoch, "steps": steps,
        "sec": round(sec, 2), "return": value, "previous": previous,
        "path": path.name,
    }
    with open(path, "wb") as f:
      pickle.dump({
          **info,
          "config": self.config,
          "params": jax.device_get(nnx.to_pure_dict(params)),
          "eval_key": np.asarray(eval_key),
          "episode": episode,
      }, f)
    with open(self.log_path, "a") as f:
      f.write(json.dumps(info) + "\n")
    return True
