"""Entry point: configuration and run setup. Training logic is in train.py.

Usage (key=value overrides, as in pgx/examples/minatar-ppo):
  uv run --package ppo python main.py                       # Breakout, 20M steps
  uv run --package ppo python main.py env=2048 seed=1
  uv run --package ppo python main.py platform=cpu num_envs=64 total_timesteps=50000

Then turn the records into a video:
  uv run --package ppo python story.py runs/minatar-breakout-seed0
"""

import json
import os
import sys
from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, field_validator

from envs import ENVS

HERE = Path(__file__).parent


class Config(BaseModel):
  model_config = ConfigDict(extra="forbid")

  env: str = "minatar-breakout"
  seed: int = 0
  platform: Literal["mps", "cpu"] = "mps"
  # PPO (defaults from pgx/examples/minatar-ppo)
  lr: float = 0.0003
  num_envs: int = 4096
  num_steps: int = 128
  total_timesteps: int = 20_000_000
  update_epochs: int = 3
  minibatch_size: int = 4096
  gamma: float = 0.99
  gae_lambda: float = 0.95
  clip_eps: float = 0.2
  ent_coef: float = 0.01
  vf_coef: float = 0.5
  max_grad_norm: float = 0.5
  # evaluation / records
  num_eval_envs: int = 100
  eval_every: int = 1  # epochs between evaluations (records are checked there)
  run_dir: str | None = None  # default: runs/<env>-seed<seed>

  @field_validator("env", mode="before")
  @classmethod
  def _known_env(cls, v):
    v = str(v)  # OmegaConf parses env=2048 as an int
    if v not in ENVS:
      raise ValueError(f"unknown env {v!r}; options: {', '.join(ENVS)}")
    return v


def main():
  cfg = Config(**OmegaConf.to_object(OmegaConf.from_cli()))
  assert (cfg.num_envs * cfg.num_steps) % cfg.minibatch_size == 0, (
      "num_envs * num_steps must be a multiple of minibatch_size"
  )
  os.environ.setdefault("JAX_PLATFORMS", cfg.platform)
  run_dir = Path(cfg.run_dir) if cfg.run_dir else HERE / "runs" / f"{cfg.env}-seed{cfg.seed}"
  run_dir.mkdir(parents=True, exist_ok=True)
  (run_dir / "config.json").write_text(cfg.model_dump_json(indent=2))
  print(json.dumps(cfg.model_dump()), file=sys.stderr)

  import train  # after JAX_PLATFORMS is set

  train.train(cfg, run_dir)
  print(f"\nrecordes em {run_dir / 'records'}; vídeo: python story.py {run_dir}")


if __name__ == "__main__":
  main()
