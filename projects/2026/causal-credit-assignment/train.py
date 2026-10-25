"""phi-PPO with Prioritized Trajectory Replay on pgx MinAtar.

Structure follows pgx/examples/minatar-ppo (itself derived from PureJaxRL:
https://github.com/luchris429/purejaxrl), with the network ported to
flax.nnx and the credit assignment of

  Li, Lee & Bareinboim. "Counterfactual Shapley Credit Assignment". RLJ 2026.

  algo=ppo      original PPO (GAE on environment rewards), the baseline.
  algo=phi_ppo  rewards replaced by counterfactual Shapley values (phi-MDP),
                plus PTR (ptr=true, Algo. 3) to replay high-|phi| chunks.

Usage:
  uv run --package causal-credit-assignment python train.py algo=phi_ppo
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Literal, NamedTuple

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, field_validator


class PPOConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  env_name: Literal[
      "minatar-breakout",
      "minatar-freeway",
      "minatar-space_invaders",
      "minatar-asterix",
      "minatar-seaquest",
      "2048",
  ] = "minatar-freeway"
  algo: Literal["ppo", "phi_ppo"] = "phi_ppo"
  platform: Literal["mps", "cpu"] = "mps"
  seed: int = 0
  lr: float = 0.0003
  num_envs: int = 4096  # B: trajectory chunks per iteration
  num_eval_envs: int = 100
  # Chunks may cross episode ends (each segment is its own Shapley game). In
  # Freeway episodes always last 2501 = 41 * 61 steps, so 61 aligns them.
  num_steps: int = 61
  total_timesteps: int = 20_000_000  # observed (fresh) environment steps
  update_epochs: int = 3
  minibatch_size: int = 4096
  gamma: float = 0.99
  gae_lambda: float = 0.95
  clip_eps: float = 0.2
  ent_coef: float = 0.01
  vf_coef: float = 0.5
  max_grad_norm: float = 0.5
  reward_scale: float = 1.0  # multiplies training rewards (2048 merges are big)
  eval_every: int = 1
  # Keep params + eval key of the best evaluation episode so far, to replay
  # and render it afterwards with render.py (no cost during training).
  save_best: bool = True
  out_dir: str = "results"
  # --- phi-PPO (Table 3) ---
  num_coalitions: int = 1  # M
  antithetic: bool = True  # pair each coalition z with 1 - z (Sec. 4)
  nte_lambda: float = 1.0  # lambda of the bootstrapped NTE (Eqs. 10-11)
  phi_lambda: float = 0.95  # lambda_phi, GAE on the phi-MDP
  phi_vf_coef: float = 0.5  # c2
  # --- PTR (Appendix B) ---
  ptr: bool = True
  buffer_mult: int = 4  # capacity C = buffer_mult * num_envs
  gamma_xi: float = 0.99
  gamma_p: float = 0.99
  eps_evict: float = 0.05
  alpha_ptr: float = 0.6
  eps_ptr: float = 0.01
  alpha_ema: float = 0.99
  dual_clip: float = 3.0  # bound on -r*A for A < 0 with replay (stability)

  @field_validator("env_name", mode="before")
  @classmethod
  def _env_name_as_str(cls, v):
    return str(v)  # OmegaConf parses env_name=2048 as an int


args = PPOConfig(**OmegaConf.to_object(OmegaConf.from_cli()))
os.environ.setdefault("JAX_PLATFORMS", args.platform)
print(args, file=sys.stderr)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import pgx  # noqa: E402
from flax import nnx  # noqa: E402
from pgx.experimental import auto_reset  # noqa: E402

from agent import eval_init, eval_step, make_forward, masked  # noqa: E402

import phi as phi_lib  # noqa: E402
import ptr as ptr_lib  # noqa: E402

env = pgx.make(str(args.env_name))
use_phi = args.algo == "phi_ppo"
use_ptr = use_phi and args.ptr
T, N = args.num_steps, args.num_envs
num_minibatches = N * T // args.minibatch_size
assert num_minibatches * args.minibatch_size == N * T


init_params, forward = make_forward(env, args.seed)


def policy_value(params, state):
  logits, value, _ = forward(params, state.observation)
  return masked(logits, state.legal_action_mask), value


def log_prob(logits, action):
  logp = jax.nn.log_softmax(logits)
  return jnp.take_along_axis(logp, action[..., None], axis=-1)[..., 0]


def entropy(logits):
  logp = jax.nn.log_softmax(logits)
  return -(jnp.exp(logp) * logp).sum(-1)


optimizer = optax.chain(
    optax.clip_by_global_norm(args.max_grad_norm), optax.adam(args.lr, eps=1e-5)
)
step_fn = jax.vmap(auto_reset(env.step, env.init))
init_fn = jax.vmap(env.init)


class Transition(NamedTuple):
  done: jnp.ndarray
  action: jnp.ndarray
  value: jnp.ndarray
  value_phi: jnp.ndarray
  reward: jnp.ndarray
  log_prob: jnp.ndarray
  obs: jnp.ndarray
  legal_mask: jnp.ndarray
  fingerprint: jnp.ndarray  # hash of the pre-step state (CTFSIM)


class Runner(NamedTuple):
  params: nnx.State
  opt_state: optax.OptState
  env_state: pgx.State
  buffer: ptr_lib.Buffer
  rng: jnp.ndarray


def rollout(params, state0, env_keys, act_key, forced_actions, use_forced):
  """Observed rollout of T steps. Replay slots (`use_forced`) re-simulate the
  stored actions; with the stored keys this reproduces the chunk exactly."""

  def _env_step(state, inp):
    k_env, k_act, a_forced = inp
    logits, value, value_phi = forward(params, state.observation)
    logits = masked(logits, state.legal_action_mask)
    action = jax.random.categorical(k_act, logits)
    action = jnp.where(use_forced, a_forced, action)
    next_state = step_fn(state, action, k_env)
    transition = Transition(
        next_state.terminated.astype(jnp.float32),
        action,
        value,
        value_phi,
        next_state.rewards[:, 0] * args.reward_scale,
        log_prob(logits, action),
        state.observation,
        state.legal_action_mask,
        phi_lib.state_fingerprint(state),
    )
    return next_state, transition

  act_keys = jax.random.split(act_key, T)
  return jax.lax.scan(
      _env_step, state0, (env_keys, act_keys, forced_actions)
  )


def gae(traj, last_val):
  def _get_advantages(carry, t):
    gae, next_value = carry
    delta = t.reward + args.gamma * next_value * (1 - t.done) - t.value
    gae = delta + args.gamma * args.gae_lambda * (1 - t.done) * gae
    return (gae, t.value), gae

  _, adv = jax.lax.scan(
      _get_advantages,
      (jnp.zeros_like(last_val), last_val),
      traj,
      reverse=True,
      unroll=16,
  )
  return adv, adv + traj.value


def shapley_credit(params, rng, state0, env_keys, traj, last_val):
  """L3EST (Algo. 1): M coalitions per chunk (plus complements if antithetic).

  Each episode segment inside the chunk is an independent Shapley game.
  """
  seg = phi_lib.segments(traj.done)
  Y = phi_lib.segment_returns(traj.reward, seg, last_val, args.gamma)

  def _credit(key, z):
    cf = phi_lib.counterfactual_rollout(
        key, state0, env_keys, traj.action, traj.fingerprint, traj.done, z,
        step_fn, init_fn, lambda state: policy_value(params, state),
    )
    cf = cf._replace(rewards=cf.rewards * args.reward_scale)
    f = phi_lib.treatment_effects(
        Y, cf, seg, traj.done, args.gamma, args.nte_lambda
    )
    w = phi_lib.shapley_weights(z, seg, args.antithetic)
    return w * f, cf.off_track.mean()

  def _one(key):
    k_z, k_cf, k_anti = jax.random.split(key, 3)
    z = phi_lib.sample_coalitions(k_z, seg)
    phi, off_track = _credit(k_cf, z)
    if args.antithetic:
      phi_anti, off_anti = _credit(k_anti, ~z)
      phi, off_track = phi + phi_anti, 0.5 * (off_track + off_anti)
    return phi, off_track

  phis, off_track = jax.vmap(_one)(jax.random.split(rng, args.num_coalitions))
  return phis.mean(0), off_track.mean()


def make_update_fn():
  def _update_step(runner: Runner, n_fresh, beta_is):
    params, opt_state, env_state, buffer, rng = runner
    rng, k_env, k_act, k_rep, k_phi, k_perm = jax.random.split(rng, 6)
    env_keys = jax.random.split(k_env, T * N).reshape(T, N, 2)

    # --- choose fresh / replay slots (Algo. 3, lines 3-10) ---
    if use_ptr:
      buffer = ptr_lib.decay(buffer, args.gamma_p, args.eps_evict)
      n_fresh = jnp.where(buffer.valid.any(), n_fresh, N)
      is_fresh = jnp.arange(N) < n_fresh
      idx = ptr_lib.sample(k_rep, buffer, N)
      replay_state = jax.tree_util.tree_map(lambda x: x[idx], buffer.state0)
      state0 = jax.tree_util.tree_map(
          lambda f, r: jnp.where(
              is_fresh.reshape((N,) + (1,) * (f.ndim - 1)), f, r
          ),
          env_state,
          replay_state,
      )
      env_keys = jnp.where(
          is_fresh[None, :, None], env_keys, buffer.keys[idx].swapaxes(0, 1)
      )
      forced = buffer.actions[idx].T
      weight = jnp.where(is_fresh, 1.0, ptr_lib.is_weights(buffer, idx, beta_is))
    else:
      n_fresh = jnp.int32(N)
      is_fresh = jnp.ones((N,), jnp.bool_)
      state0 = env_state
      forced = jnp.zeros((T, N), jnp.int32)
      weight = jnp.ones((N,))

    # --- observed rollout ---
    final_state, traj = rollout(
        params, state0, env_keys, k_act, forced, ~is_fresh
    )
    env_state = jax.tree_util.tree_map(
        lambda f, o: jnp.where(
            is_fresh.reshape((N,) + (1,) * (f.ndim - 1)), f, o
        ),
        final_state,
        env_state,
    )
    _, last_val, last_val_phi = forward(params, final_state.observation)
    advantages, targets = gae(traj, last_val)

    metrics = {"n_fresh": n_fresh}
    if use_phi:
      phi, diverged = shapley_credit(
          params, k_phi, state0, env_keys, traj, last_val
      )
      phi_adv, phi_targets = phi_lib.phi_gae(
          phi, traj.value_phi, traj.done, last_val_phi, args.phi_lambda
      )
      # normalize by std only: phi = 0 stays the no-effect baseline
      actor_adv = phi_adv / (phi_adv.std() + 1e-8)
      metrics.update(
          phi_abs=jnp.abs(phi).mean(), phi_sum=phi.sum(0).mean(),
          off_track=diverged,
      )
      if use_ptr:
        is_new = is_fresh
        ema_prev = jnp.where(is_new[None], 0.0, buffer.ema_phi2[idx].T)
        n_prev = jnp.where(is_new, 0, buffer.n_updates[idx])
        ema, n_upd, _, p_j = ptr_lib.priorities(
            phi, ema_prev, n_prev, args.alpha_ema, args.alpha_ptr, args.eps_ptr
        )
        p_bar = ptr_lib.expected_priority(buffer)
        wins = (p_j > p_bar) & is_fresh
        metrics.update(
            win_rate=wins.sum() / jnp.maximum(is_fresh.sum(), 1),
            buffer_size=buffer.valid.sum(),
        )
        buffer = ptr_lib.update(
            buffer, is_fresh, idx, state0, env_keys.swapaxes(0, 1),
            traj.action.T, ema.T, n_upd, p_j,
        )
    else:
      actor_adv = advantages
      phi_targets = jnp.zeros_like(targets)

    # --- PPO update (Eq. 23) ---
    w = jnp.broadcast_to(weight[None], (T, N))

    def _loss_fn(params, batch):
      traj, adv, targets, phi_targets, w = batch
      logits, value, value_phi = forward(params, traj.obs)
      logits = masked(logits, traj.legal_mask)
      logp = log_prob(logits, traj.action)

      value_pred_clipped = traj.value + (value - traj.value).clip(
          -args.clip_eps, args.clip_eps
      )
      value_loss = 0.5 * jnp.maximum(
          jnp.square(value - targets), jnp.square(value_pred_clipped - targets)
      ).mean()

      log_ratio = logp - traj.log_prob
      if use_ptr:
        # replayed actions can have ~0 probability under the current policy
        log_ratio = jnp.clip(log_ratio, -20.0, 20.0)
      ratio = jnp.exp(log_ratio)
      if not use_phi:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
      loss_actor = -jnp.minimum(
          ratio * adv,
          jnp.clip(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv,
      )
      if use_ptr:
        # dual-clip PPO (Ye et al., 2020): with A < 0 the clipped objective
        # does not bound r * A, which blew up to NaN once the policy became
        # near-deterministic and replayed old actions had r >> 1.
        loss_actor = jnp.where(
            adv < 0, jnp.minimum(loss_actor, -args.dual_clip * adv), loss_actor
        )
      loss_actor = (w * loss_actor).mean()
      ent = entropy(logits).mean()

      total = loss_actor + args.vf_coef * value_loss - args.ent_coef * ent
      phi_loss = 0.0
      if use_phi:
        phi_loss = 0.5 * jnp.square(value_phi - phi_targets).mean()
        total = total + args.phi_vf_coef * phi_loss
      return total, (value_loss, phi_loss, loss_actor, ent)

    def _update_minibatch(carry, batch):
      params, opt_state = carry
      (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
          params, batch
      )
      updates, opt_state = optimizer.update(grads, opt_state)
      params = optax.apply_updates(params, updates)
      return (params, opt_state), (loss, *aux)

    batch = (traj, actor_adv, targets, phi_targets, w)
    batch = jax.tree_util.tree_map(
        lambda x: x.reshape((N * T,) + x.shape[2:]), batch
    )

    def _update_epoch(carry, key):
      perm = jax.random.permutation(key, N * T)
      minibatches = jax.tree_util.tree_map(
          lambda x: jnp.take(x, perm, axis=0).reshape(
              (num_minibatches, -1) + x.shape[1:]
          ),
          batch,
      )
      return jax.lax.scan(_update_minibatch, carry, minibatches)

    (params, opt_state), losses = jax.lax.scan(
        _update_epoch,
        (params, opt_state),
        jax.random.split(k_perm, args.update_epochs),
    )
    loss, value_loss, phi_loss, actor_loss, ent = jax.tree_util.tree_map(
        jnp.mean, losses
    )
    metrics.update(
        loss=loss, value_loss=value_loss, phi_loss=phi_loss,
        actor_loss=actor_loss, entropy=ent,
    )
    runner = Runner(params, opt_state, env_state, buffer, rng)
    return runner, metrics

  return _update_step


@jax.jit
def evaluate(params, rng_key):
  """Returns the return of each evaluation episode, shape (num_eval_envs,)."""
  state, rng_key = eval_init(env, rng_key, args.num_eval_envs)
  R = jnp.zeros_like(state.rewards)

  def cond_fn(tup):
    state, _, _ = tup
    return ~state.terminated.all()

  def loop_fn(tup):
    state, R, rng_key = tup
    state, rng_key = eval_step(env, forward, params, state, rng_key)
    return state, R + state.rewards, rng_key

  _, R, _ = jax.lax.while_loop(cond_fn, loop_fn, (state, R, rng_key))
  return R[:, 0]


def train(rng):
  out_dir = Path(__file__).parent / args.out_dir
  out_dir.mkdir(parents=True, exist_ok=True)
  run_name = f"{args.env_name}-{args.algo}-seed{args.seed}"
  log_path = out_dir / f"{run_name}.jsonl"
  log_file = open(log_path, "w")
  best_path = out_dir / "best" / f"{run_name}.pkl"
  best_return = -np.inf

  rng, k_reset = jax.random.split(rng)
  env_state = jax.jit(jax.vmap(env.init))(jax.random.split(k_reset, N))
  single_state = jax.tree_util.tree_map(lambda x: x[0], env_state)
  capacity = args.buffer_mult * N if use_ptr else 1
  buffer = ptr_lib.init_buffer(single_state, capacity, T)
  rng, k_runner = jax.random.split(rng)
  runner = Runner(
      init_params, optimizer.init(init_params), env_state, buffer, k_runner
  )
  update_step = jax.jit(make_update_fn())

  # Thompson sampling over the fresh fraction xi (Eq. 15)
  np_rng = np.random.default_rng(args.seed)
  alpha = beta = 0.0
  observed_steps = sim_steps = 0

  def log(i, metrics, sec):
    nonlocal best_return
    rng_eval = jax.random.fold_in(rng, i)
    returns = np.asarray(evaluate(runner.params, rng_eval))
    rec = {
        "iter": i, "sec": round(sec, 2), "observed_steps": observed_steps,
        "sim_steps": sim_steps, "eval_R": float(returns.mean()),
        "eval_R_max": float(returns.max()),
        **{k: float(v) for k, v in metrics.items()},
    }
    if args.save_best and returns.max() > best_return:
      best_return = float(returns.max())
      best_path.parent.mkdir(exist_ok=True)
      with open(best_path, "wb") as f:
        pickle.dump({
            "config": args.model_dump(),
            "params": jax.device_get(nnx.to_pure_dict(runner.params)),
            "eval_key": np.asarray(rng_eval),
            "episode": int(returns.argmax()),
            "return": best_return,
            "iter": i,
            "observed_steps": observed_steps,
        }, f)
    print(rec, flush=True)
    log_file.write(json.dumps(rec) + "\n")
    log_file.flush()

  train_sec = 0.0  # excludes evaluation time, as in the pgx example
  log(0, {}, train_sec)
  st = time.time()
  i = 0
  while observed_steps < args.total_timesteps:
    i += 1
    xi = np_rng.beta(alpha + 1, beta + 1)
    n_fresh = int(np.clip(np.floor(xi * N), 1, N - 1)) if use_ptr else N
    beta_is = min(1.0, observed_steps / args.total_timesteps)
    runner, metrics = update_step(runner, jnp.int32(n_fresh), beta_is)
    n_fresh = int(metrics["n_fresh"])
    observed_steps += n_fresh * T
    num_cf = args.num_coalitions * (2 if args.antithetic else 1)
    sim_steps += N * T * (1 + (num_cf if use_phi else 0))
    if use_ptr:
      u = float(metrics["win_rate"])
      alpha = args.gamma_xi * alpha + u
      beta = args.gamma_xi * beta + (1 - u)
      metrics["xi"] = xi
    if i % args.eval_every == 0 or observed_steps >= args.total_timesteps:
      jax.block_until_ready(runner.params)
      train_sec += time.time() - st
      log(i, metrics, train_sec)
      st = time.time()
  log_file.close()
  print(f"log: {log_path}", file=sys.stderr)
  return runner


if __name__ == "__main__":
  train(jax.random.PRNGKey(args.seed))
