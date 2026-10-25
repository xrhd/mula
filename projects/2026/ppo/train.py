"""PPO from scratch on a pgx single-player game.

Modified from pgx/examples/minatar-ppo, itself derived from PureJaxRL
(https://github.com/luchris429/purejaxrl). The network is in flax.nnx and the
whole update (rollout -> GAE -> minibatch epochs) is one jitted function.

Configuration lives in main.py; this module only knows how to train.
"""

import json
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pgx
from pgx.experimental import auto_reset

from agent import eval_init, eval_step, make_forward, masked
from envs import ENVS
from recorder import RecordKeeper


class Transition(NamedTuple):
  done: jnp.ndarray
  action: jnp.ndarray
  value: jnp.ndarray
  reward: jnp.ndarray
  log_prob: jnp.ndarray
  obs: jnp.ndarray
  legal_mask: jnp.ndarray


def log_prob(logits, action):
  logp = jax.nn.log_softmax(logits)
  return jnp.take_along_axis(logp, action[..., None], axis=-1)[..., 0]


def entropy(logits):
  logp = jax.nn.log_softmax(logits)
  return -(jnp.exp(logp) * logp).sum(-1)


def make_update_step(env, forward, optimizer, cfg, reward_scale):
  """One PPO epoch: collect num_envs x num_steps, compute GAE, then
  update_epochs passes of minibatch SGD on the clipped objective."""
  step_fn = jax.vmap(auto_reset(env.step, env.init))
  batch_size = cfg.num_envs * cfg.num_steps
  num_minibatches = batch_size // cfg.minibatch_size

  def _env_step(carry, _):
    params, env_state, rng = carry
    rng, k_act, k_env = jax.random.split(rng, 3)
    logits, value = forward(params, env_state.observation)
    logits = masked(logits, env_state.legal_action_mask)
    action = jax.random.categorical(k_act, logits)
    next_state = step_fn(env_state, action, jax.random.split(k_env, cfg.num_envs))
    transition = Transition(
        done=next_state.terminated.astype(jnp.float32),
        action=action,
        value=value,
        reward=next_state.rewards[:, 0] * reward_scale,
        log_prob=log_prob(logits, action),
        obs=env_state.observation,
        legal_mask=env_state.legal_action_mask,
    )
    return (params, next_state, rng), transition

  def _gae(traj, last_value):
    def _step(carry, t):
      gae, next_value = carry
      delta = t.reward + cfg.gamma * next_value * (1 - t.done) - t.value
      gae = delta + cfg.gamma * cfg.gae_lambda * (1 - t.done) * gae
      return (gae, t.value), gae

    _, adv = jax.lax.scan(
        _step, (jnp.zeros_like(last_value), last_value), traj, reverse=True,
        unroll=16,
    )
    return adv, adv + traj.value

  def _loss(params, batch):
    traj, adv, targets = batch
    logits, value = forward(params, traj.obs)
    logits = masked(logits, traj.legal_mask)
    logp = log_prob(logits, traj.action)

    # clipped value loss
    value_clipped = traj.value + (value - traj.value).clip(
        -cfg.clip_eps, cfg.clip_eps
    )
    value_loss = 0.5 * jnp.maximum(
        jnp.square(value - targets), jnp.square(value_clipped - targets)
    ).mean()

    # clipped policy loss with normalized advantages
    ratio = jnp.exp(logp - traj.log_prob)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    actor_loss = -jnp.minimum(
        ratio * adv, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
    ).mean()
    ent = entropy(logits).mean()

    total = actor_loss + cfg.vf_coef * value_loss - cfg.ent_coef * ent
    return total, (value_loss, actor_loss, ent)

  def _minibatch(carry, batch):
    params, opt_state = carry
    (loss, aux), grads = jax.value_and_grad(_loss, has_aux=True)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    return (optax.apply_updates(params, updates), opt_state), (loss, *aux)

  def update_step(params, opt_state, env_state, rng):
    # 1) rollout
    (params, env_state, rng), traj = jax.lax.scan(
        _env_step, (params, env_state, rng), None, cfg.num_steps
    )
    # 2) advantages
    _, last_value = forward(params, env_state.observation)
    adv, targets = _gae(traj, last_value)
    # 3) update_epochs passes over shuffled minibatches
    batch = jax.tree_util.tree_map(
        lambda x: x.reshape((batch_size,) + x.shape[2:]), (traj, adv, targets)
    )

    def _epoch(carry, key):
      perm = jax.random.permutation(key, batch_size)
      minibatches = jax.tree_util.tree_map(
          lambda x: x[perm].reshape((num_minibatches, -1) + x.shape[1:]), batch
      )
      return jax.lax.scan(_minibatch, carry, minibatches)

    rng, k_perm = jax.random.split(rng)
    (params, opt_state), losses = jax.lax.scan(
        _epoch, (params, opt_state), jax.random.split(k_perm, cfg.update_epochs)
    )
    loss, value_loss, actor_loss, ent = jax.tree_util.tree_map(jnp.mean, losses)
    metrics = {
        "loss": loss, "value_loss": value_loss, "actor_loss": actor_loss,
        "entropy": ent,
    }
    return params, opt_state, env_state, rng, metrics

  return update_step


def make_evaluate(env, forward, num_envs: int):
  """Plays `num_envs` full episodes; returns the return of each one."""

  @jax.jit
  def evaluate(params, rng_key):
    state, rng_key = eval_init(env, rng_key, num_envs)
    R = jnp.zeros(num_envs)

    def cond(carry):
      return ~carry[0].terminated.all()

    def body(carry):
      state, R, rng_key = carry
      state, rng_key = eval_step(env, forward, params, state, rng_key)
      return state, R + state.rewards[:, 0], rng_key

    _, R, _ = jax.lax.while_loop(cond, body, (state, R, rng_key))
    return R

  return evaluate


def train(cfg, run_dir: Path):
  """Trains PPO and logs metrics.jsonl + record checkpoints in `run_dir`."""
  spec = ENVS[cfg.env]
  env = pgx.make(spec.pgx_id)
  run_dir.mkdir(parents=True, exist_ok=True)

  params, forward = make_forward(env, cfg.seed)
  optimizer = optax.chain(
      optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr, eps=1e-5)
  )
  opt_state = optimizer.init(params)
  update_step = jax.jit(
      make_update_step(env, forward, optimizer, cfg, spec.reward_scale)
  )
  evaluate = make_evaluate(env, forward, cfg.num_eval_envs)
  recorder = RecordKeeper(run_dir, cfg.model_dump())

  rng = jax.random.PRNGKey(cfg.seed)
  rng, k_reset, k_train = jax.random.split(rng, 3)
  env_state = jax.jit(jax.vmap(env.init))(jax.random.split(k_reset, cfg.num_envs))

  steps_per_epoch = cfg.num_envs * cfg.num_steps
  num_epochs = cfg.total_timesteps // steps_per_epoch
  metrics_file = open(run_dir / "metrics.jsonl", "w")
  train_sec = 0.0

  def log(epoch, metrics):
    eval_key = jax.random.fold_in(rng, epoch)
    returns = np.asarray(evaluate(params, eval_key))
    steps = epoch * steps_per_epoch
    new_record = recorder.update(returns, params, eval_key, epoch, steps, train_sec)
    rec = {
        "epoch": epoch, "steps": steps, "sec": round(train_sec, 2),
        "eval_mean": float(returns.mean()), "eval_max": float(returns.max()),
        "record": recorder.best, "new_record": new_record,
        **{k: float(v) for k, v in metrics.items()},
    }
    metrics_file.write(json.dumps(rec) + "\n")
    metrics_file.flush()
    flag = "  ★ novo recorde" if new_record else ""
    print(
        f"época {epoch:4d}/{num_epochs} | {steps / 1e6:6.2f}M passos |"
        f" {train_sec:7.1f}s | eval {rec['eval_mean']:8.2f} (máx"
        f" {rec['eval_max']:.0f}){flag}",
        flush=True,
    )

  log(0, {})  # the random initial policy is record #1
  for epoch in range(1, num_epochs + 1):
    st = time.time()
    params, opt_state, env_state, k_train, metrics = update_step(
        params, opt_state, env_state, k_train
    )
    jax.block_until_ready(params)
    train_sec += time.time() - st  # evaluation time excluded, as in pgx
    if epoch % cfg.eval_every == 0 or epoch == num_epochs:
      log(epoch, metrics)

  metrics_file.close()
  return params
