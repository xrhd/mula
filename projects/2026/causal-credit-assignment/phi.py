"""Counterfactual Shapley credit assignment (Li, Lee & Bareinboim, RLJ 2026).

Pieces of the paper implemented here, all vectorized over a batch of `N`
trajectory chunks of length `T` (time-major arrays, shape `(T, N, ...)`):

  * `segments`: episode segments inside a chunk. Each segment is its own
    Shapley game whose players are the actions of that segment.
  * `coalition_probs` / `sample_coalitions`: optimal proposal Q* (Thm. 4.2).
  * `shapley_weights`: kappa_t(z) / Q*(z) in closed form (Def. 4.1).
  * `counterfactual_rollout`: CTFSIM (Algo. 2) with common random numbers.
  * `segment_returns` / `treatment_effects`: NTE per segment (Eqs. 4, 10-11).
  * `phi_gae`: phi-advantages on the undiscounted phi-MDP (Eqs. 12-13).

In pgx all the exogenous noise of a step comes from the PRNG key passed to
`env.step`, so the key at step t *is* U_t in the MDP-SCM (Def. 2.2).
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


class Segments(NamedTuple):
  """Per-step description of the episode segment containing each step."""

  start: jnp.ndarray  # (T, N) index of the first step of the segment
  end: jnp.ndarray  # (T, N) index of the last step of the segment
  length: jnp.ndarray  # (T, N) number of steps (players) in the segment
  open: jnp.ndarray  # (T, N) segment reaches the chunk end without a done


def segments(dones) -> Segments:
  """Splits each chunk column at the observed dones (done at t ends a segment)."""
  T, N = dones.shape
  d = dones.astype(jnp.bool_)
  t = jnp.broadcast_to(jnp.arange(T)[:, None], (T, N))
  prev = jnp.concatenate([jnp.zeros((1, N), jnp.bool_), d[:-1]])
  start = jax.lax.cummax(jnp.where(prev | (t == 0), t, 0), axis=0)
  end = jax.lax.cummin(jnp.where(d, t, T - 1), axis=0, reverse=True)
  is_open = ~jnp.take_along_axis(d, end, axis=0)
  return Segments(start, end, end - start + 1, is_open)


def _exclusive_cumsum(x):
  return jnp.concatenate([jnp.zeros_like(x[:1]), jnp.cumsum(x, axis=0)])


def segment_sum(x, seg: Segments):
  """Sum of `x` over the segment of each step, broadcast to `(T, N)`."""
  E = _exclusive_cumsum(x)
  return jnp.take_along_axis(E, seg.end + 1, 0) - jnp.take_along_axis(
      E, seg.start, 0
  )


def sum_since_start(x, seg: Segments):
  """sum_{start <= j < t} x_j for every step t."""
  E = _exclusive_cumsum(x)
  return E[:-1] - jnp.take_along_axis(E, seg.start, 0)


def _c(k, L):
  """c_k of Eq. 9 for a game with L players (k in 1..L)."""
  k = k.astype(jnp.float32)
  L = L.astype(jnp.float32)
  inner = (1.0 / L) * (1.0 / jnp.maximum(k, 1.0) + 1.0 / jnp.maximum(L - k, 1.0))
  return jnp.where(k < L, inner, 1.0 / L**2)


def _q(k, L, T: int):
  """q*_k ∝ sqrt(c_k) for a game with L <= T players."""
  ks = jnp.arange(1, T + 1)[None, :]
  Ls = jnp.arange(1, T + 1)[:, None]
  norm = jnp.where(ks <= Ls, jnp.sqrt(_c(ks, Ls)), 0.0).sum(1)  # (T,)
  return jnp.sqrt(_c(k, L)) / norm[L - 1]


def coalition_probs(T: int) -> jnp.ndarray:
  """q*_k for k = 1..T (index k-1) in a single game of T players."""
  return _q(jnp.arange(1, T + 1), jnp.full((T,), T), T)


def sample_coalitions(key, seg: Segments) -> jnp.ndarray:
  """Samples one coalition z ~ Q* per segment, returning bool `(T, N)`.

  First the size k ~ q*_k (for the segment length L), then a uniformly random
  subset of size k (Q*(z) = q*_k / C(L, k)). k = 0 is never sampled since
  f(0) = Y - Y = 0.
  """
  T = seg.start.shape[0]
  key_k, key_u = jax.random.split(key)
  ks = jnp.arange(1, T + 1)
  L = seg.length[..., None]
  logq = jnp.where(ks <= L, jnp.log(_q(ks, jnp.minimum(L, T), T)), -jnp.inf)
  k = jax.random.categorical(key_k, logq) + 1  # (T, N), one per step
  k = jnp.take_along_axis(k, seg.start, 0)  # use the draw of the 1st step
  u = jax.random.uniform(key_u, seg.start.shape)
  order = seg.start.astype(jnp.float32) + u  # sorts by segment, then by u
  rank = jnp.argsort(jnp.argsort(order, axis=0), axis=0)
  return (rank - seg.start) < k


def shapley_weights(z, seg: Segments, antithetic: bool = False):
  """kappa_t(z) / Q*(z) for masks `z` of shape `(T, N)`, per segment.

  With Q*(z) = q*_k / C(L, k) and the Shapley kernel of Eq. 5 the binomials
  cancel:  w_t = 1 / (k q*_k) if z_t = 1, and -1 / ((L - k) q*_k) otherwise.

  With `antithetic=True` the weights are for a pair {z, 1 - z} (Covert & Lee,
  2021): a mask m lands in the pair with probability Q*(m) + Q*(1 - m), so
  sum_{m in pair} kappa(m) f(m) / (Q*(m) + Q*(1 - m)) stays unbiased. As
  q*_k = q*_{L-k} for 0 < k < L this halves the weights; the full coalition
  (k = L) keeps its weight and the empty one (k = 0, f = 0) gets none.
  """
  T = z.shape[0]
  L = seg.length
  k = segment_sum(z.astype(jnp.int32), seg)
  qk = _q(jnp.clip(k, 1, L), L, T)
  pos = 1.0 / (jnp.maximum(k, 1) * qk)
  neg = -1.0 / (jnp.maximum(L - k, 1) * qk)
  w = jnp.where(z, pos, neg)
  if antithetic:
    w = jnp.where((k > 0) & (k < L), 0.5 * w, w)
  return jnp.where(k > 0, w, 0.0)


def _discount_since_start(seg: Segments, gamma):
  T = seg.start.shape[0]
  t = jnp.arange(T)[:, None]
  return gamma ** (t - seg.start).astype(jnp.float32)


def segment_returns(rewards, seg: Segments, last_value, gamma):
  """Observed return of each segment, broadcast to its steps.

  Y = sum_{t in seg} gamma^(t - start) r_t, plus gamma^L V(s_T) for the open
  (last) segment of the chunk.
  """
  disc = _discount_since_start(seg, gamma)
  Y = segment_sum(disc * rewards, seg)
  boot = gamma ** seg.length.astype(jnp.float32) * last_value[None]
  return Y + jnp.where(seg.open, boot, 0.0)


def state_fingerprint(state) -> jnp.ndarray:
  """64-bit hash `(N, 2)` of a batched pgx state, used to test s^z_t == s_t.

  Every leaf except `rewards` (the last reward does not drive the dynamics)
  is bit-cast to uint32 and projected with fixed pseudo-random multipliers
  (wrap-around arithmetic). Collisions are negligible for this use.
  """
  paths_leaves = jax.tree_util.tree_flatten_with_path(state)[0]
  n = paths_leaves[0][1].shape[0]
  h = jnp.zeros((n, 2), jnp.uint32)
  for i, (path, leaf) in enumerate(paths_leaves):
    if "rewards" in jax.tree_util.keystr(path):
      continue
    x = leaf.reshape(n, -1)
    if jnp.issubdtype(x.dtype, jnp.floating):
      x = jax.lax.bitcast_convert_type(x.astype(jnp.float32), jnp.uint32)
    x = x.astype(jnp.uint32)
    mult = jax.random.bits(
        jax.random.PRNGKey(i), (2, x.shape[1]), jnp.uint32
    ) | jnp.uint32(1)
    h = h + jnp.stack([(x * mult[0]).sum(1), (x * mult[1]).sum(1)], axis=1)
  return h


class Counterfactual(NamedTuple):
  rewards: jnp.ndarray  # (T, N)
  dones: jnp.ndarray  # (T, N)
  values: jnp.ndarray  # (T, N) V_psi(s^z_t)
  last_value: jnp.ndarray  # (N,) V_psi(s^z_T)
  off_track: jnp.ndarray  # (T, N) whether s^z_t != s_t


def counterfactual_rollout(
    key,
    state0,
    env_keys: jnp.ndarray,
    actions: jnp.ndarray,
    fingerprints: jnp.ndarray,
    obs_dones: jnp.ndarray,
    z: jnp.ndarray,
    step_fn: Callable,
    init_fn: Callable,
    policy_value_fn: Callable,
) -> Counterfactual:
  """CTFSIM (Algo. 2) with self-baseline (pi_base = pi).

  Args:
    key: PRNG key for the counterfactual action samples.
    state0: batched pgx state at the start of the chunk.
    env_keys: `(T, N, 2)` keys used by the observed trajectory (U_t).
    actions: `(T, N)` observed actions x_t.
    fingerprints: `(T, N, 2)` `state_fingerprint` of the observed s_t.
    obs_dones: `(T, N)` observed dones (segment boundaries).
    z: `(T, N)` coalition masks (True = intervened).
    step_fn: batched auto-reset `step(state, action, keys)`.
    init_fn: batched `env.init(keys)`.
    policy_value_fn: `state -> (logits, value)` (logits already masked).

  When s^z_t = s_t and z_t = 0 the observed action is reused; otherwise a
  fresh action is drawn from pi (pi_base = pi when z_t = 1). When
  (s^z_t, x^z_t) = (s_t, x_t) the observed key is reused, so (s_{t+1}, y_t)
  is reproduced exactly; otherwise the step gets fresh noise. Comparing
  states at every step lets the counterfactual rejoin the track (Fig. 3):
  an intervention without effect on the state gets phi = 0.

  When the observed episode ends, the counterfactual is moved to the same
  initial state of the next episode (the reset is exogenous: pgx's auto-reset
  draws it from the observed key), so every segment is its own game.
  """
  T = actions.shape[0]

  def _step(state, inp):
    x_t, fp_t, d_t, z_t, k_env, k_act = inp
    on_track = (state_fingerprint(state) == fp_t).all(-1)
    logits, value = policy_value_fn(state)
    resampled = jax.random.categorical(k_act, logits)
    a = jnp.where(on_track & ~z_t, x_t, resampled)
    same_pair = on_track & (a == x_t)
    fresh_keys = jax.vmap(lambda k: jax.random.fold_in(k, 1))(k_env)
    keys = jnp.where(same_pair[:, None], k_env, fresh_keys)
    state = step_fn(state, a, keys)
    out = (state.rewards[:, 0], state.terminated, value, ~on_track)
    # auto_reset draws the next episode from split(key)[1]: mirror it
    reset_keys = jax.vmap(lambda k: jax.random.split(k)[1])(k_env)
    reset = init_fn(reset_keys).replace(terminated=jnp.ones_like(d_t))
    state = jax.tree_util.tree_map(
        lambda r, s: jnp.where(d_t.reshape((-1,) + (1,) * (s.ndim - 1)), r, s),
        reset,
        state,
    )
    return state, out

  act_keys = jax.random.split(key, T)
  state, (rewards, dones, values, off_track) = jax.lax.scan(
      _step,
      state0,
      (actions, fingerprints, obs_dones.astype(jnp.bool_), z, env_keys,
       act_keys),
  )
  _, last_value = policy_value_fn(state)
  return Counterfactual(
      rewards, dones.astype(jnp.float32), values, last_value, off_track
  )


def treatment_effects(Y, cf: Counterfactual, seg: Segments, obs_dones, gamma,
                      lam):
  """Per-step lambda-bootstrapped NTE (Eqs. 10-11), inside each segment.

  f^z_t = Y - (R^z_t + gamma^(t - start) G^{z,lam}_t), with Y the observed
  segment return, R^z_t the exact counterfactual rewards of the segment
  before t and G^{z,lam}_t the counterfactual lambda-return from t. If the
  counterfactual episode ends before the observed one, its later rewards
  are dropped. With lam = 1 every f_t equals the exact NTE Y - Y^z.
  """
  next_values = jnp.concatenate([cf.values[1:], cf.last_value[None]])
  cut = jnp.maximum(cf.dones, obs_dones.astype(jnp.float32))

  def _lam_return(G_next, inp):
    r, d, v_next = inp
    G = r + gamma * (1.0 - d) * ((1.0 - lam) * v_next + lam * G_next)
    return G, G

  _, G = jax.lax.scan(
      _lam_return, cf.last_value, (cf.rewards, cut, next_values), reverse=True
  )
  alive = (sum_since_start(cf.dones, seg) == 0).astype(jnp.float32)
  disc = _discount_since_start(seg, gamma)
  R_before = sum_since_start(disc * alive * cf.rewards, seg)
  return Y - (R_before + disc * alive * G)


def phi_gae(phi, values, dones, last_value, lam):
  """phi-advantages (Eq. 13) on the undiscounted phi-MDP (gamma_phi = 1).

  delta_t = phi_t + V^phi(s_{t+1}) - V^phi(s_t). Returns (advantages,
  targets) with targets = A + V^phi used to regress the phi-critic.
  """

  def _gae(carry, inp):
    gae, next_value = carry
    p, v, d = inp
    delta = p + next_value * (1.0 - d) - v
    gae = delta + lam * (1.0 - d) * gae
    return (gae, v), gae

  _, adv = jax.lax.scan(
      _gae,
      (jnp.zeros_like(last_value), last_value),
      (phi, values, dones),
      reverse=True,
  )
  return adv, adv + values
