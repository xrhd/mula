"""Actor-critic network (flax.nnx) and evaluation episode, shared by
train.py and render.py so a saved evaluation can be replayed exactly."""

import jax
import jax.numpy as jnp
from flax import nnx


class Head(nnx.Module):

  def __init__(self, out: int, rngs: nnx.Rngs):
    self.l1 = nnx.Linear(64, 64, rngs=rngs)
    self.l2 = nnx.Linear(64, 64, rngs=rngs)
    self.l3 = nnx.Linear(64, out, rngs=rngs)

  def __call__(self, x):
    x = jnp.tanh(self.l1(x))
    x = jnp.tanh(self.l2(x))
    return self.l3(x)


class ActorCritic(nnx.Module):
  """Same torso as the pgx example, with a third head for V^phi."""

  def __init__(self, obs_shape, num_actions: int, rngs: nnx.Rngs):
    h, w, c = obs_shape
    self.conv = nnx.Conv(c, 32, kernel_size=(2, 2), rngs=rngs)
    self.torso = nnx.Linear((h // 2) * (w // 2) * 32, 64, rngs=rngs)
    self.actor = Head(num_actions, rngs)
    self.critic = Head(1, rngs)  # V_psi: original MDP (bootstrapping)
    self.phi_critic = Head(1, rngs)  # V^phi_omega: phi-MDP (advantages)

  def __call__(self, x):
    x = x.astype(jnp.float32)
    x = jax.nn.relu(self.conv(x))
    x = nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
    x = x.reshape((x.shape[0], -1))
    x = jax.nn.relu(self.torso(x))
    v = self.critic(x)[..., 0]
    v_phi = self.phi_critic(x)[..., 0]
    return self.actor(x), v, v_phi


def masked(logits, legal_action_mask):
  """Illegal actions (e.g. 2048) get zero probability."""
  return jnp.where(legal_action_mask, logits, jnp.finfo(logits.dtype).min)


def make_forward(env, seed: int = 0):
  """Returns (init_params, forward) for `env`."""
  model = ActorCritic(env.observation_shape, env.num_actions, nnx.Rngs(seed))
  graphdef, init_params = nnx.split(model)

  def forward(params, obs):
    return nnx.merge(graphdef, params)(obs)

  return init_params, forward


def eval_init(env, rng_key, num_envs: int):
  """First half of an evaluation: (state, rng_key) for `num_envs` episodes."""
  rng_key, sub_key = jax.random.split(rng_key)
  state = jax.vmap(env.init)(jax.random.split(sub_key, num_envs))
  return state, rng_key


def eval_step(env, forward, params, state, rng_key):
  """One step of every evaluation episode (actions sampled from pi)."""
  logits, _, _ = forward(params, state.observation)
  logits = masked(logits, state.legal_action_mask)
  rng_key, k_act, k_env = jax.random.split(rng_key, 3)
  action = jax.random.categorical(k_act, logits)
  keys = jax.random.split(k_env, state.observation.shape[0])
  return jax.vmap(env.step)(state, action, keys), rng_key
