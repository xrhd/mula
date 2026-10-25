"""Prioritized Trajectory Replay (PTR), Appendix B of the paper.

The buffer stores *how to re-simulate* a chunk instead of its observations:
the pgx state at the start of the chunk, the step keys (exogenous noise) and
the observed actions. pgx steps are deterministic given (state, action, key),
so replaying a chunk reproduces it exactly, and phi-values are recomputed
with the current policy and value function.

Everything has static shapes (capacity `C`, chunk length `T`) so the buffer
lives inside the jitted update.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


class Buffer(NamedTuple):
  state0: Any  # batched pgx State, leading dim C
  keys: jnp.ndarray  # (C, T, 2) step keys
  actions: jnp.ndarray  # (C, T)
  ema_phi2: jnp.ndarray  # (C, T) EMA of phi^2 (not bias corrected)
  n_updates: jnp.ndarray  # (C,) number of EMA updates
  priority: jnp.ndarray  # (C,) trajectory priority p_j = max_t p_{j,t}
  valid: jnp.ndarray  # (C,)


def init_buffer(state_example, capacity: int, T: int) -> Buffer:
  """Empty buffer; `state_example` is a single (unbatched) pgx state."""
  state0 = jax.tree_util.tree_map(
      lambda x: jnp.zeros((capacity,) + x.shape, x.dtype), state_example
  )
  return Buffer(
      state0=state0,
      keys=jnp.zeros((capacity, T, 2), jnp.uint32),
      actions=jnp.zeros((capacity, T), jnp.int32),
      ema_phi2=jnp.zeros((capacity, T), jnp.float32),
      n_updates=jnp.zeros((capacity,), jnp.int32),
      priority=jnp.zeros((capacity,), jnp.float32),
      valid=jnp.zeros((capacity,), jnp.bool_),
  )


def decay(buf: Buffer, gamma_p: float, eps_evict: float) -> Buffer:
  """p_j <- gamma_p p_j, evicting entries with p_j < eps_evict."""
  priority = buf.priority * gamma_p
  valid = buf.valid & (priority >= eps_evict)
  return buf._replace(priority=jnp.where(valid, priority, 0.0), valid=valid)


def sampling_probs(buf: Buffer) -> jnp.ndarray:
  """P_j = p_j / sum_i p_i over valid entries (uniform if empty)."""
  p = jnp.where(buf.valid, buf.priority, 0.0)
  total = p.sum()
  uniform = jnp.full_like(p, 1.0 / p.shape[0])
  return jnp.where(total > 0, p / jnp.maximum(total, 1e-12), uniform)


def sample(key, buf: Buffer, n: int):
  """Samples `n` indices proportional to priority."""
  probs = sampling_probs(buf)
  return jax.random.choice(key, probs.shape[0], (n,), p=probs)


def is_weights(buf: Buffer, idx, beta_is):
  """w_j = (|B| P_j)^-beta / max_j' (|B| P_j')^-beta  (Eq. 18)."""
  probs = sampling_probs(buf)
  size = jnp.maximum(buf.valid.sum(), 1)
  min_p = jnp.min(jnp.where(buf.valid, probs, jnp.inf))
  min_p = jnp.where(jnp.isfinite(min_p), min_p, 1.0)
  w = (size * probs[idx]) ** (-beta_is)
  w_max = (size * min_p) ** (-beta_is)
  return w / w_max


def priorities(phi, ema_prev, n_prev, alpha_ema, alpha_ptr, eps_ptr):
  """Eq. 20 on a batch: phi `(T, N)`, ema_prev `(T, N)`, n_prev `(N,)`.

  Returns (ema, n, p_t, p_j) where p_t = (bias-corrected EMA of phi^2)^
  (alpha_ptr / 2) + eps_ptr and p_j = max_t p_t.
  """
  ema = alpha_ema * ema_prev + (1.0 - alpha_ema) * phi**2
  n = n_prev + 1
  corrected = ema / (1.0 - alpha_ema ** n.astype(jnp.float32))
  p_t = corrected ** (alpha_ptr / 2.0) + eps_ptr
  return ema, n, p_t, p_t.max(axis=0)


def expected_priority(buf: Buffer):
  """p̄_B = sum p^2 / sum p, the expected priority under proportional draws."""
  p = jnp.where(buf.valid, buf.priority, 0.0)
  return (p**2).sum() / jnp.maximum(p.sum(), 1e-12)


def update(
    buf: Buffer,
    is_fresh,
    idx,
    state0,
    keys,
    actions,
    ema,
    n,
    p_j,
) -> Buffer:
  """Refreshes replayed entries in place and inserts fresh chunks.

  Batch arrays are slot-major: state0 `(N, ...)`, keys `(N, T, 2)`, actions,
  ema `(N, T)`, n and p_j `(N,)`. Fresh chunks overwrite empty or
  lowest-priority entries. Masked writes use out-of-bounds indices with
  `mode="drop"`.
  """
  C = buf.priority.shape[0]
  N = is_fresh.shape[0]

  # 1) Replayed entries: new EMA statistics and priority.
  ridx = jnp.where(is_fresh, C, idx)
  buf = buf._replace(
      ema_phi2=buf.ema_phi2.at[ridx].set(ema, mode="drop"),
      n_updates=buf.n_updates.at[ridx].set(n, mode="drop"),
      priority=buf.priority.at[ridx].set(p_j, mode="drop"),
  )

  # 2) Fresh chunks go to empty slots first, then to the lowest priorities.
  score = jnp.where(buf.valid, buf.priority, -1.0)
  _, slots = jax.lax.top_k(-score, N)
  fidx = jnp.where(is_fresh, slots, C)

  def _set(dst, src):
    return dst.at[fidx].set(src, mode="drop")

  return Buffer(
      state0=jax.tree_util.tree_map(_set, buf.state0, state0),
      keys=_set(buf.keys, keys),
      actions=_set(buf.actions, actions),
      ema_phi2=_set(buf.ema_phi2, ema),
      n_updates=_set(buf.n_updates, n),
      priority=_set(buf.priority, p_j),
      valid=_set(buf.valid, jnp.ones((N,), jnp.bool_)),
  )
