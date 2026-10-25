"""Actor-critic network (flax.nnx) and the evaluation episode.

`eval_init` / `eval_step` are shared by train.py and render.py: evaluation is
deterministic given (params, key), so a record can be replayed exactly.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class ActorCritic(nnx.Module):
  """Same architecture as pgx/examples/minatar-ppo (there in haiku)."""

  def __init__(self, obs_shape, num_actions: int, rngs: nnx.Rngs):
    h, w, c = obs_shape
    self.conv = nnx.Conv(c, 32, kernel_size=(2, 2), rngs=rngs)
    self.torso = nnx.Linear((h // 2) * (w // 2) * 32, 64, rngs=rngs)
    self.actor = nnx.List([
        nnx.Linear(64, 64, rngs=rngs),
        nnx.Linear(64, 64, rngs=rngs),
        nnx.Linear(64, num_actions, rngs=rngs),
    ])
    self.critic = nnx.List([
        nnx.Linear(64, 64, rngs=rngs),
        nnx.Linear(64, 64, rngs=rngs),
        nnx.Linear(64, 1, rngs=rngs),
    ])

  def __call__(self, x):
    x = x.astype(jnp.float32)
    x = jax.nn.relu(self.conv(x))
    x = nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
    x = x.reshape((x.shape[0], -1))
    x = jax.nn.relu(self.torso(x))
    return _mlp(self.actor, x), _mlp(self.critic, x)[..., 0]


def _mlp(layers, x):
  for layer in layers[:-1]:
    x = jnp.tanh(layer(x))
  return layers[-1](x)


def masked(logits, legal_action_mask):
  """Illegal actions (e.g. in 2048) get zero probability."""
  return jnp.where(legal_action_mask, logits, jnp.finfo(logits.dtype).min)


def make_forward(env, seed: int = 0):
  """Returns (params, forward) where forward(params, obs) -> (logits, value)."""
  model = ActorCritic(env.observation_shape, env.num_actions, nnx.Rngs(seed))
  graphdef, params = nnx.split(model)

  def forward(params, obs):
    return nnx.merge(graphdef, params)(obs)

  return params, forward


def eval_init(env, rng_key, num_envs: int):
  """Starts `num_envs` evaluation episodes."""
  rng_key, sub_key = jax.random.split(rng_key)
  state = jax.vmap(env.init)(jax.random.split(sub_key, num_envs))
  return state, rng_key


def eval_step(env, forward, params, state, rng_key):
  """One step of every evaluation episode, actions sampled from pi."""
  logits, _ = forward(params, state.observation)
  logits = masked(logits, state.legal_action_mask)
  rng_key, k_act, k_env = jax.random.split(rng_key, 3)
  action = jax.random.categorical(k_act, logits)
  keys = jax.random.split(k_env, state.observation.shape[0])
  return jax.vmap(env.step)(state, action, keys), rng_key
