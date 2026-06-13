import datetime
import os
import pickle
import time
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import mctx
import numpy as np
import optax
import pgx
from flax import nnx
from matplotlib.animation import PillowWriter
from omegaconf import OmegaConf
from pgx.experimental import auto_reset
from pydantic import BaseModel

from mula.models.alphazero.modeling import AZNet
from mula.models.alphazero.params import ModelConfig

matplotlib.use("Agg")

devices = jax.local_devices()
num_devices = len(devices)


def _maybe_pmap(f):
    """Use pmap on multi-device, jit on single-device."""
    if num_devices > 1:
        return jax.pmap(f)
    else:
        return jax.jit(f)


class Config(BaseModel):
    env_id: pgx.EnvId = "othello"
    seed: int = 0
    max_num_iters: int = 400
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    selfplay_batch_size: int = 1024
    num_simulations: int = 32
    max_num_steps: int = 256
    training_batch_size: int = 4096
    learning_rate: float = 0.001
    eval_interval: int = 5

    class Config:
        extra = "ignore"


config: Config | None = None
env = None
baseline = None
optimizer = None


def _init_config(default_config: Config | None = None):
    global config, env, baseline, optimizer
    if default_config is None:
        conf_dict = OmegaConf.from_cli()
        config = Config(**conf_dict)
    else:
        config = default_config
    print(config)
    env = pgx.make(config.env_id)
    try:
        baseline = pgx.make_baseline_model(config.env_id + "_v0")
    except Exception as e:
        print(f"Warning: could not load baseline model: {e}")
        baseline = None
    optimizer = optax.adam(learning_rate=config.learning_rate)


def _ensure_config():
    if config is None:
        _init_config(Config())


def make_model(rngs: nnx.Rngs):
    _ensure_config()
    return AZNet(
        num_actions=env.num_actions,
        num_channels=config.num_channels,
        num_blocks=config.num_layers,
        resnet_v2=config.resnet_v2,
        rngs=rngs,
    )


def recurrent_fn(params_state, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
    """MCTS recurrent function for Gumbel MuZero policy."""
    _ensure_config()
    del rng_key
    params, state_bn = params_state

    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)

    model = nnx.merge(*params_state)
    logits, value = model(state.observation, use_running_average=True)
    graphdef, new_state = nnx.split(model)
    del graphdef

    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(
        state.legal_action_mask, logits, jnp.finfo(logits.dtype).min
    )

    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    value = jnp.where(state.terminated, 0.0, value)
    discount = -1.0 * jnp.ones_like(value)
    discount = jnp.where(state.terminated, 0.0, discount)

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return recurrent_fn_output, state


class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray


def _selfplay_impl(params_state, rng_key: jnp.ndarray) -> SelfplayOutput:
    """Core selfplay logic (not pmapped)."""
    _ensure_config()
    batch_size = config.selfplay_batch_size

    def step_fn(state, key) -> SelfplayOutput:
        key1, key2 = jax.random.split(key)
        observation = state.observation

        model = nnx.merge(*params_state)
        logits, value = model(state.observation, use_running_average=True)
        graphdef, _ = nnx.split(model)
        del graphdef

        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

        policy_output = mctx.gumbel_muzero_policy(
            params=params_state,
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        actor = state.current_player
        keys = jax.random.split(key2, batch_size)
        state = jax.vmap(auto_reset(env.step, env.init))(
            state, policy_output.action, keys
        )
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)
        return state, SelfplayOutput(
            obs=observation,
            action_weights=policy_output.action_weights,
            reward=state.rewards[jnp.arange(state.rewards.shape[0]), actor],
            terminated=state.terminated,
            discount=discount,
        )

    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)
    key_seq = jax.random.split(rng_key, config.max_num_steps)
    _, data = jax.lax.scan(step_fn, state, key_seq)
    return data


selfplay = _maybe_pmap(_selfplay_impl)


class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


def _compute_loss_input_impl(data: SelfplayOutput) -> Sample:
    """Core loss input computation (not pmapped)."""
    _ensure_config()
    batch_size = config.selfplay_batch_size
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    def body_fn(carry, i):
        ix = config.max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v

    _, value_tgt = jax.lax.scan(
        body_fn,
        jnp.zeros(batch_size),
        jnp.arange(config.max_num_steps),
    )
    value_tgt = value_tgt[::-1, :]

    return Sample(
        obs=data.obs,
        policy_tgt=data.action_weights,
        value_tgt=value_tgt,
        mask=value_mask,
    )


compute_loss_input = _maybe_pmap(_compute_loss_input_impl)


def loss_fn(model, samples: Sample):
    logits, value = model(samples.obs, use_running_average=False)

    policy_loss = optax.softmax_cross_entropy(logits, samples.policy_tgt)
    policy_loss = jnp.mean(policy_loss)

    value_loss = optax.l2_loss(value, samples.value_tgt)
    value_loss = jnp.mean(value_loss * samples.mask)

    return policy_loss + value_loss, (policy_loss, value_loss)


def _train_step_impl(model, opt_state, data: Sample):
    _ensure_config()
    grads, (policy_loss, value_loss) = nnx.grad(loss_fn, has_aux=True)(
        model, data
    )
    if num_devices > 1:
        grads = jax.lax.pmean(grads, axis_name="i")
    updates, opt_state = optimizer.update(grads, opt_state)
    params = nnx.state(model, nnx.Param)
    params = optax.apply_updates(params, updates)
    nnx.update(model, params)
    return model, opt_state, policy_loss, value_loss


train_step = _maybe_pmap(_train_step_impl)


def _evaluate_impl(rng_key, my_model):
    """Simplified evaluation by sampling against baseline."""
    _ensure_config()
    my_player = 0

    key, subkey = jax.random.split(rng_key)
    batch_size = config.selfplay_batch_size // num_devices
    keys = jax.random.split(subkey, batch_size)
    state = jax.vmap(env.init)(keys)

    def body_fn(val):
        key, state, R = val
        my_logits, _ = my_model(state.observation, use_running_average=True)
        if baseline is not None:
            opp_logits, _ = baseline(state.observation)
        else:
            # Random opponent if baseline not available
            opp_logits = jnp.zeros_like(my_logits)
        is_my_turn = (state.current_player == my_player).reshape((-1, 1))
        logits = jnp.where(is_my_turn, my_logits, opp_logits)
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, logits, axis=-1)
        state = jax.vmap(env.step)(state, action)
        R = R + state.rewards[jnp.arange(batch_size), my_player]
        return (key, state, R)

    _, _, R = jax.lax.while_loop(
        lambda x: ~(x[1].terminated.all()),
        body_fn,
        (key, state, jnp.zeros(batch_size)),
    )
    return R


evaluate = _maybe_pmap(_evaluate_impl)


def _elo_from_win_rate(win_rate: float, k: int = 400) -> float:
    """Convert win rate to ELO difference."""
    if win_rate <= 0:
        return -k
    if win_rate >= 1:
        return k
    return -k * np.log10(1 / win_rate - 1)


def _plot_training_curves(
    iteration_history: list,
    policy_loss_history: list,
    value_loss_history: list,
    elo_history: list,
    plot_dir: str,
    gif_path: str,
):
    """Plot training curves and generate an animated GIF."""
    os.makedirs(plot_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def _update(frame):
        for ax in axes:
            ax.clear()

        iters = iteration_history[: frame + 1]
        p_loss = policy_loss_history[: frame + 1]
        v_loss = value_loss_history[: frame + 1]
        elo = elo_history[: frame + 1]

        axes[0].plot(iters, p_loss, label="policy loss", color="blue")
        axes[0].plot(iters, v_loss, label="value loss", color="orange")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(iters, elo, label="ELO", color="green")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("ELO")
        axes[1].set_title("ELO vs Baseline")
        axes[1].legend()
        axes[1].grid(True)

        fig.tight_layout()
        return axes

    writer = PillowWriter(fps=2)
    with writer.saving(fig, gif_path, dpi=100):
        for i in range(len(iteration_history)):
            _update(i)
            writer.grab_frame()

    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(
        iteration_history, policy_loss_history, label="policy loss", color="blue"
    )
    axes[0].plot(
        iteration_history, value_loss_history, label="value loss", color="orange"
    )
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(iteration_history, elo_history, label="ELO", color="green")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("ELO")
    axes[1].set_title("ELO vs Baseline")
    axes[1].legend()
    axes[1].grid(True)
    fig.tight_layout()
    png_path = os.path.join(plot_dir, "training_curves.png")
    fig.savefig(png_path)
    plt.close(fig)
    print(f"Saved plots to {plot_dir} and GIF to {gif_path}")


if __name__ == "__main__":
    _init_config()  # parse CLI args
    rng_key = jax.random.PRNGKey(config.seed)
    model_rngs = nnx.Rngs(params=rng_key)
    model = make_model(model_rngs)

    # Split into graphdef + state for pmap compat
    graphdef, state = nnx.split(model)
    params = nnx.state(model, nnx.Param)
    opt_state = optimizer.init(params)

    if num_devices > 1:
        graphdef = jax.device_put_replicated(graphdef, devices)
        state = jax.device_put_replicated(state, devices)
        opt_state = jax.device_put_replicated(opt_state, devices)

    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    now_str = now.strftime("%Y%m%d%H%M%S")
    ckpt_dir = os.path.join("checkpoints", f"{config.env_id}_{now_str}")
    plot_dir = os.path.join("plots", f"{config.env_id}_{now_str}")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    iteration = 0
    hours = 0.0
    frames = 0
    log = {"iteration": iteration, "hours": hours, "frames": frames}

    iteration_history: list[int] = []
    policy_loss_history: list[float] = []
    value_loss_history: list[float] = []
    elo_history: list[float] = []

    rng_key = jax.random.PRNGKey(config.seed)
    while True:
        if iteration % config.eval_interval == 0:
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)
            if num_devices == 1:
                keys = keys[0]
            # Merge model for eval
            eval_model = nnx.merge(graphdef, state)
            R = evaluate(keys, eval_model)
            win_rate = ((R == 1).sum() / R.size).item()
            draw_rate = ((R == 0).sum() / R.size).item()
            lose_rate = ((R == -1).sum() / R.size).item()
            elo = _elo_from_win_rate(win_rate)

            log.update(
                {
                    "eval/vs_baseline/avg_R": R.mean().item(),
                    "eval/vs_baseline/win_rate": win_rate,
                    "eval/vs_baseline/draw_rate": draw_rate,
                    "eval/vs_baseline/lose_rate": lose_rate,
                    "eval/vs_baseline/elo": elo,
                }
            )

            iteration_history.append(iteration)
            policy_loss_history.append(
                policy_loss_history[-1] if policy_loss_history else 0.0
            )
            value_loss_history.append(
                value_loss_history[-1] if value_loss_history else 0.0
            )
            elo_history.append(elo)

            # Save checkpoint
            ckpt_model = nnx.merge(
                jax.tree_util.tree_map(lambda x: x[0], graphdef) if num_devices > 1 else graphdef,
                jax.tree_util.tree_map(lambda x: x[0], state) if num_devices > 1 else state,
            )
            ckpt_opt_state = (
                jax.tree_util.tree_map(lambda x: x[0], opt_state)
                if num_devices > 1
                else opt_state
            )
            with open(os.path.join(ckpt_dir, f"{iteration:06d}.ckpt"), "wb") as f:
                dic = {
                    "config": config.model_dump(),
                    "rng_key": rng_key,
                    "model": jax.device_get(ckpt_model),
                    "opt_state": jax.device_get(ckpt_opt_state),
                    "iteration": iteration,
                    "frames": frames,
                    "hours": hours,
                    "pgx.__version__": pgx.__version__,
                    "env_id": env.id,
                    "env_version": env.version,
                }
                pickle.dump(dic, f)

        print(log)

        if iteration >= config.max_num_iters:
            break

        iteration += 1
        log = {"iteration": iteration}
        st = time.time()

        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, num_devices)
        if num_devices == 1:
            keys = keys[0]
        params_state = (graphdef, state)
        data: SelfplayOutput = selfplay(params_state, keys)
        samples: Sample = compute_loss_input(data)

        samples = jax.device_get(samples)
        if num_devices == 1:
            # Add dummy device dimension to match original code expectations
            samples = jax.tree_util.tree_map(lambda x: x[None, ...], samples)
        frames += (
            samples.obs.shape[0]
            * samples.obs.shape[1]
            * samples.obs.shape[2]
        )
        samples = jax.tree_util.tree_map(
            lambda x: x.reshape((-1, *x.shape[3:])), samples
        )
        rng_key, subkey = jax.random.split(rng_key)
        ixs = jax.random.permutation(subkey, jnp.arange(samples.obs.shape[0]))
        samples = jax.tree_util.tree_map(lambda x: x[ixs], samples)
        num_updates = samples.obs.shape[0] // config.training_batch_size
        if num_devices > 1:
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape((num_updates, num_devices, -1) + x.shape[1:]),
                samples,
            )
        else:
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape((num_updates, -1) + x.shape[1:]),
                samples,
            )

        policy_losses, value_losses = [], []
        for i in range(num_updates):
            minibatch: Sample = jax.tree_util.tree_map(lambda x: x[i], minibatches)
            # Merge model for training
            train_model = nnx.merge(graphdef, state)
            train_model, opt_state, policy_loss, value_loss = train_step(
                train_model, opt_state, minibatch
            )
            # Split back
            graphdef, state = nnx.split(train_model)
            policy_losses.append(policy_loss.mean().item())
            value_losses.append(value_loss.mean().item())
        policy_loss = sum(policy_losses) / len(policy_losses)
        value_loss = sum(value_losses) / len(value_losses)

        et = time.time()
        hours += (et - st) / 3600
        log.update(
            {
                "train/policy_loss": policy_loss,
                "train/value_loss": value_loss,
                "hours": hours,
                "frames": frames,
            }
        )

        if iteration_history and iteration_history[-1] == iteration:
            policy_loss_history[-1] = policy_loss
            value_loss_history[-1] = value_loss
        else:
            iteration_history.append(iteration)
            policy_loss_history.append(policy_loss)
            value_loss_history.append(value_loss)
            elo_history.append(elo_history[-1] if elo_history else 0.0)

    gif_path = os.path.join(plot_dir, "training_evolution.gif")
    _plot_training_curves(
        iteration_history,
        policy_loss_history,
        value_loss_history,
        elo_history,
        plot_dir,
        gif_path,
    )
