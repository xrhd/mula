"""Tests for the phi-value estimator, CTFSIM and the PTR buffer (CPU)."""

import itertools
import math
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import pgx  # noqa: E402
from pgx.experimental import auto_reset  # noqa: E402

import phi as phi_lib  # noqa: E402
import ptr as ptr_lib  # noqa: E402


def exact_shapley(f, T):
  """Eq. 3 by brute force; f maps a tuple mask to a value with f(0) = 0."""
  out = np.zeros(T)
  for t in range(T):
    for z in itertools.product([0, 1], repeat=T):
      if z[t]:
        continue
      k = sum(z)
      w = math.factorial(k) * math.factorial(T - k - 1) / math.factorial(T)
      with_t = tuple(1 if s == t else zs for s, zs in enumerate(z))
      out[t] += w * (f[with_t] - f[z])
  return out


def _one_game(T, n=1):
  return phi_lib.segments(jnp.zeros((T, n)))


def _weights(z, antithetic=False):
  z = jnp.array(z, dtype=bool)[:, None]
  return np.asarray(phi_lib.shapley_weights(z, _one_game(len(z)), antithetic))[:, 0]


def test_estimator_is_unbiased_against_exact_shapley():
  T = 4
  rng = np.random.default_rng(0)
  masks = list(itertools.product([0, 1], repeat=T))
  f = {z: (0.0 if sum(z) == 0 else rng.normal()) for z in masks}
  q = np.asarray(phi_lib.coalition_probs(T))

  expected = np.zeros(T)
  for z in masks:
    k = sum(z)
    if k == 0:
      continue
    Qz = q[k - 1] / math.comb(T, k)
    expected += Qz * _weights(z) * f[z]

  np.testing.assert_allclose(expected, exact_shapley(f, T), rtol=1e-5)


def test_antithetic_pair_estimator_is_unbiased():
  T = 5
  rng = np.random.default_rng(1)
  masks = list(itertools.product([0, 1], repeat=T))
  f = {z: (0.0 if sum(z) == 0 else rng.normal()) for z in masks}
  q = np.asarray(phi_lib.coalition_probs(T))

  expected = np.zeros(T)
  for z in masks:  # z ~ Q*, the estimate uses both z and its complement
    k = sum(z)
    if k == 0:
      continue
    Qz = q[k - 1] / math.comb(T, k)
    zc = tuple(1 - x for x in z)
    for m in (z, zc):
      expected += Qz * _weights(m, antithetic=True) * f[m]

  np.testing.assert_allclose(expected, exact_shapley(f, T), rtol=1e-5)


def test_weights_sum_matches_efficiency():
  T = 7
  q = np.asarray(phi_lib.coalition_probs(T))
  for k in range(1, T + 1):
    total = float(_weights(np.arange(T) < k).sum())
    target = 1.0 / q[T - 1] if k == T else 0.0
    assert abs(total - target) < 1e-3 * max(1.0, target)


def test_sample_coalitions_sizes_follow_q():
  T, n = 10, 50_000
  z = phi_lib.sample_coalitions(jax.random.PRNGKey(0), _one_game(T, n))
  k = np.asarray(z.sum(0))
  assert k.min() >= 1 and k.max() <= T
  freq = np.bincount(k, minlength=T + 1)[1:] / n
  np.testing.assert_allclose(freq, np.asarray(phi_lib.coalition_probs(T)), atol=0.01)


def test_segments_split_at_dones_and_games_are_independent():
  dones = jnp.array([[0, 1, 0, 0, 1, 0, 0]], jnp.float32).T  # segments 2, 3, 2
  seg = phi_lib.segments(dones)
  np.testing.assert_array_equal(np.asarray(seg.start[:, 0]), [0, 0, 2, 2, 2, 5, 5])
  np.testing.assert_array_equal(np.asarray(seg.length[:, 0]), [2, 2, 3, 3, 3, 2, 2])
  np.testing.assert_array_equal(np.asarray(seg.open[:, 0]), [0, 0, 0, 0, 0, 1, 1])

  # every sampled coalition is non-empty inside every segment
  z = phi_lib.sample_coalitions(jax.random.PRNGKey(0), phi_lib.segments(jnp.tile(dones, (1, 2000))))
  z = np.asarray(z)
  for lo, hi in [(0, 2), (2, 5), (5, 7)]:
    assert (z[lo:hi].sum(0) >= 1).all()

  # the weights of a segment only depend on that segment's mask
  zz = jnp.array([[1, 0, 0, 1, 1, 1, 0]], bool).T
  w = np.asarray(phi_lib.shapley_weights(zz, seg))[:, 0]
  np.testing.assert_allclose(w[2:5], _weights([0, 1, 1]), rtol=1e-6)
  np.testing.assert_allclose(w[5:7], _weights([1, 0]), rtol=1e-6)


def test_segment_returns():
  r = jnp.array([[1.0], [1.0], [1.0]])
  d = jnp.array([[0.0], [1.0], [0.0]])
  seg = phi_lib.segments(d)
  Y = phi_lib.segment_returns(r, seg, jnp.array([100.0]), 0.5)
  # first segment: 1 + 0.5; second (open): 1 + 0.5 * 100
  np.testing.assert_allclose(np.asarray(Y[:, 0]), [1.5, 1.5, 51.0])


def _fingerprints(step_fn, state0, actions, env_keys):
  def _step(state, inp):
    a, k = inp
    return step_fn(state, a, k), phi_lib.state_fingerprint(state)

  return jax.lax.scan(_step, state0, (actions, env_keys))[1]


def _chunk(T=50, N=8, seed=0, env_id="minatar-freeway"):
  env = pgx.make(env_id)
  step_fn = jax.vmap(auto_reset(env.step, env.init))
  key = jax.random.PRNGKey(seed)
  k_init, k_env, k_act = jax.random.split(key, 3)
  state0 = jax.vmap(env.init)(jax.random.split(k_init, N))
  env_keys = jax.random.split(k_env, T * N).reshape(T, N, 2)
  actions = jax.random.randint(k_act, (T, N), 0, env.num_actions)

  def _step(state, inp):
    a, k = inp
    state = step_fn(state, a, k)
    return state, (state.rewards[:, 0], state.terminated.astype(jnp.float32))

  _, (rewards, dones) = jax.lax.scan(_step, state0, (actions, env_keys))
  return env, step_fn, state0, env_keys, actions, rewards, dones


def _uniform_policy(num_actions):
  def policy_value(state):
    obs = state.observation
    n = obs.shape[0]
    v = obs.reshape(n, -1).sum(-1).astype(jnp.float32) * 0.01
    return jnp.zeros((n, num_actions)), v

  return policy_value


def test_ctfsim_empty_coalition_reproduces_observed_trajectory():
  # Breakout episodes end at different times, so chunks cross resets.
  T, N = 200, 8
  env, step_fn, state0, env_keys, actions, rewards, dones = _chunk(
      T, N, env_id="minatar-breakout"
  )
  assert int(dones.sum()) > N  # several episode ends inside the chunk
  fps = _fingerprints(step_fn, state0, actions, env_keys)
  z = jnp.zeros((T, N), dtype=bool)
  cf = phi_lib.counterfactual_rollout(
      jax.random.PRNGKey(1), state0, env_keys, actions, fps, dones, z,
      step_fn, jax.vmap(env.init), _uniform_policy(env.num_actions),
  )
  np.testing.assert_array_equal(np.asarray(cf.rewards), np.asarray(rewards))
  np.testing.assert_array_equal(np.asarray(cf.dones), np.asarray(dones))
  assert not bool(cf.off_track.any())


def test_ctfsim_resyncs_at_observed_episode_end():
  # Intervening on every action diverges, but each new observed episode starts
  # the counterfactual from the observed reset state again (on-track).
  T, N = 200, 8
  env, step_fn, state0, env_keys, actions, rewards, dones = _chunk(
      T, N, env_id="minatar-breakout"
  )
  fps = _fingerprints(step_fn, state0, actions, env_keys)
  z = jnp.ones((T, N), dtype=bool)
  cf = phi_lib.counterfactual_rollout(
      jax.random.PRNGKey(1), state0, env_keys, actions, fps, dones, z,
      step_fn, jax.vmap(env.init), _uniform_policy(env.num_actions),
  )
  seg = phi_lib.segments(dones)
  first = (np.asarray(seg.start) == np.arange(T)[:, None]) & (np.arange(T)[:, None] > 0)
  off = np.asarray(cf.off_track)
  assert first.sum() > N
  assert not off[first].any()  # back on track at every new observed episode
  assert off.mean() > 0.3  # but intervening everywhere does leave the track


@pytest.mark.parametrize("env_id", ["minatar-freeway", "minatar-breakout"])
def test_treatment_effects_lambda_one_is_exact_nte(env_id):
  T, N = 120, 6
  env, step_fn, state0, env_keys, actions, rewards, dones = _chunk(
      T, N, env_id=env_id
  )
  fps = _fingerprints(step_fn, state0, actions, env_keys)
  seg = phi_lib.segments(dones)
  z = phi_lib.sample_coalitions(jax.random.PRNGKey(3), seg)
  pv = _uniform_policy(env.num_actions)
  cf = phi_lib.counterfactual_rollout(
      jax.random.PRNGKey(4), state0, env_keys, actions, fps, dones, z,
      step_fn, jax.vmap(env.init), pv,
  )
  last = jnp.ones(N)
  Y = phi_lib.segment_returns(rewards, seg, last, 0.9)
  f = np.asarray(phi_lib.treatment_effects(Y, cf, seg, dones, 0.9, 1.0))

  # brute force per segment: counterfactual return truncated at its own done
  r, d, cd = map(np.asarray, (cf.rewards, dones, cf.dones))
  start, end = np.asarray(seg.start), np.asarray(seg.end)
  for n in range(N):
    for t in range(T):
      s, e = start[t, n], end[t, n]
      Yz, alive = 0.0, True
      for j in range(s, e + 1):
        if alive:
          Yz += 0.9 ** (j - s) * r[j, n]
        alive = alive and not cd[j, n]
      if alive and not d[e, n]:  # open segment, counterfactual still alive
        Yz += 0.9 ** (e - s + 1) * float(cf.last_value[n])
      assert abs(f[t, n] - (float(Y[t, n]) - Yz)) < 1e-4


def test_freeway_episodes_end_on_chunk_boundaries():
  # Episodes last 2501 = 41 * 61 steps, so 61-step chunks never cross a reset.
  T, N = 61, 4
  _, _, _, _, _, _, dones = _chunk(T * 41 * 2, N)
  for j in range(N):
    t_done = np.nonzero(np.asarray(dones)[:, j])[0]
    assert list(t_done) == [2500, 5001]
    assert all((t + 1) % T == 0 for t in t_done)


def _toy_buffer(C=6, T=3):
  state = {"x": jnp.zeros((2,), jnp.int32)}
  return ptr_lib.init_buffer(state, C, T)


def test_buffer_insert_decay_evict_and_is_weights():
  C, T, N = 6, 3, 4
  buf = _toy_buffer(C, T)
  is_fresh = jnp.array([True, True, True, False])
  idx = jnp.zeros((N,), jnp.int32)
  state0 = {"x": jnp.arange(N * 2, dtype=jnp.int32).reshape(N, 2)}
  keys = jnp.zeros((N, T, 2), jnp.uint32)
  actions = jnp.ones((N, T), jnp.int32)
  phi = jnp.array([[1.0, 0.1, 2.0, 9.0]] * T)
  ema, n, _, p_j = ptr_lib.priorities(
      phi, jnp.zeros((T, N)), jnp.zeros((N,), jnp.int32), 0.99, 0.6, 0.01
  )
  np.testing.assert_allclose(np.asarray(p_j), np.abs(np.asarray(phi[0])) ** 0.6 + 0.01, rtol=1e-5)

  buf = ptr_lib.update(buf, is_fresh, idx, state0, keys, actions, ema.T, n, p_j)
  assert int(buf.valid.sum()) == 3  # only fresh chunks are inserted
  stored = sorted(np.asarray(buf.priority[buf.valid]).tolist())
  np.testing.assert_allclose(stored, sorted(np.asarray(p_j[:3]).tolist()), rtol=1e-6)

  # IS weights: the least likely entry gets weight 1, more likely ones less
  valid_idx = jnp.nonzero(buf.valid, size=3)[0]
  w = np.asarray(ptr_lib.is_weights(buf, valid_idx, 1.0))
  p = np.asarray(buf.priority[valid_idx])
  assert np.isclose(w[np.argmin(p)], 1.0)
  assert np.all(w <= 1.0 + 1e-6)

  # decay + eviction removes the low-priority entry (p ≈ 0.26)
  buf = ptr_lib.decay(buf, 0.5, 0.2)
  assert int(buf.valid.sum()) == 2

  # sampling only returns valid entries
  s = np.asarray(ptr_lib.sample(jax.random.PRNGKey(0), buf, 256))
  assert np.all(np.asarray(buf.valid)[s])

