"""Tests for the PPO project (CPU)."""

import json
import os
import subprocess
import sys

os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pgx  # noqa: E402
import pytest  # noqa: E402

from ppo import render  # noqa: E402
from ppo.agent import make_forward, masked  # noqa: E402
from ppo.envs import ENVS  # noqa: E402
from ppo.main import Config  # noqa: E402
from ppo.recorder import RecordKeeper  # noqa: E402
from ppo.train import make_evaluate  # noqa: E402


@pytest.mark.parametrize("name", list(ENVS))
def test_every_registered_game_steps(name):
  env = pgx.make(ENVS[name].pgx_id)
  params, forward = make_forward(env)
  state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
  logits, value = forward(params, state.observation)
  assert logits.shape == (2, env.num_actions) and value.shape == (2,)
  state = jax.vmap(env.step)(state, jnp.zeros(2, jnp.int32), jax.random.split(jax.random.PRNGKey(1), 2))
  assert state.rewards.shape == (2, 1)


def test_mask_zeroes_illegal_actions():
  logits = jnp.zeros((1, 4))
  mask = jnp.array([[True, False, True, False]])
  p = jax.nn.softmax(masked(logits, mask))
  np.testing.assert_allclose(np.asarray(p), [[0.5, 0.0, 0.5, 0.0]], atol=1e-6)


def test_config_rejects_unknown_env_and_parses_2048():
  assert Config(env=2048).env == "2048"
  with pytest.raises(ValueError):
    Config(env="pong")


def _evaluate_and_record(tmp_path, name, num_eval_envs=6):
  env = pgx.make(ENVS[name].pgx_id)
  params, forward = make_forward(env, seed=3)
  cfg = Config(env=name, platform="cpu", num_eval_envs=num_eval_envs).model_dump()
  keeper = RecordKeeper(tmp_path, cfg)
  key = jax.random.PRNGKey(42)
  returns, _ = make_evaluate(env, forward, num_eval_envs)(params, key)
  assert keeper.update(returns, params, key, epoch=5, steps=1000, sec=1.0)
  return keeper, np.asarray(returns)


def test_recorder_only_saves_improvements(tmp_path):
  keeper, returns = _evaluate_and_record(tmp_path, "minatar-breakout")
  params = make_forward(pgx.make("minatar-breakout"))[0]
  key = jax.random.PRNGKey(0)
  assert not keeper.update(returns, params, key, 6, 2000, 2.0)  # tie
  assert keeper.update(returns + 1, params, key, 7, 3000, 3.0)
  log = [json.loads(l) for l in (tmp_path / "records.jsonl").open()]
  assert [r["record"] for r in log] == [1, 2]
  assert log[1]["previous"] == log[0]["return"]
  assert sorted(p.name for p in (tmp_path / "records").iterdir()) == [
      "001_epoca-0005.pkl", "002_epoca-0007.pkl",
  ]


@pytest.mark.parametrize("name", ["minatar-breakout", "2048"])
def test_replay_reproduces_saved_record(tmp_path, name):
  _, returns = _evaluate_and_record(tmp_path, name)
  record = render.load_record(next((tmp_path / "records").iterdir()))
  states, total = render.replay(record)
  assert total == pytest.approx(float(returns.max()))
  assert bool(states[-1].terminated)


def test_story_builds_complete_clips(tmp_path):
  run = tmp_path / "run"
  _evaluate_and_record(run, "minatar-breakout")
  cfg = Config(env="minatar-breakout", platform="cpu", num_eval_envs=6)
  (run / "config.json").write_text(cfg.model_dump_json())
  (run / "metrics.jsonl").write_text(
      json.dumps({"epoch": 0, "eval_mean": 0.5}) + "\n"
      + json.dumps({"epoch": 5, "eval_mean": 1.0}) + "\n"
  )
  out = subprocess.run(
      [sys.executable, "-m", "ppo.story", str(run), "--card-seconds", "0.1"],
      capture_output=True, text=True, env={**os.environ, "JAX_PLATFORMS": "cpu"},
  )
  assert out.returncode == 0, out.stderr
  assert (run / "story.mp4").stat().st_size > 0
  clip = json.loads((run / "story_manifest.json").read_text())[0]
  assert clip["frames"] == clip["episode_steps"] + 1  # the whole episode


def test_2048_stops_at_target_tile(tmp_path):
  from ppo.train import train

  cfg = Config(
      env="2048", platform="cpu", num_envs=16, num_steps=8, minibatch_size=128,
      total_timesteps=128 * 4, num_eval_envs=4, target_tile=16,
  )
  train(cfg, tmp_path)
  log = [json.loads(l) for l in (tmp_path / "metrics.jsonl").open()]
  assert all(r["max_tile"] >= 2 for r in log)
  assert log[-1]["max_tile"] >= 16 or len(log) == 5  # stopped at the target or ran out
  assert all(r["max_tile"] < 16 for r in log[:-1])  # never continued past it
