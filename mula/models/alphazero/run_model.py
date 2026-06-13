import argparse
import os
import pickle

import jax
import jax.numpy as jnp
import pgx
from flax import nnx

from mula.models.alphazero.modeling import AZNet
from mula.models.alphazero.params import ModelConfig


def run_model(checkpoint_dir: str | None = None, max_steps: int | None = None):
    """Run AlphaZero model to play Othello.

    If checkpoint_dir is provided, loads the latest checkpoint.
    Otherwise, uses a freshly initialized model.
    max_steps limits the number of game steps for quick testing.
    """
    env = pgx.make("othello")
    config = ModelConfig()

    rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
    model = AZNet(
        num_actions=env.num_actions,
        num_channels=config.num_channels,
        num_blocks=config.num_layers,
        resnet_v2=config.resnet_v2,
        rngs=rngs,
    )

    if checkpoint_dir is not None and os.path.isdir(checkpoint_dir):
        ckpt_files = sorted(
            [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
        )
        if ckpt_files:
            latest_ckpt = os.path.join(checkpoint_dir, ckpt_files[-1])
            print(f"Loading checkpoint: {latest_ckpt}")
            with open(latest_ckpt, "rb") as f:
                dic = pickle.load(f)
            loaded = dic.get("model")
            if loaded is not None and isinstance(loaded, nnx.Module):
                # Use the loaded model directly (same architecture)
                model = loaded
            else:
                print("Checkpoint model format not recognized, using fresh model")
        else:
            print("No checkpoint found, using fresh model")
    else:
        print("No checkpoint directory provided, using fresh model")

    # Run a single self-play game and print the board states
    key = jax.random.PRNGKey(0)
    state = env.init(key)
    print("\nStarting Othello game (model vs random):\n")
    step = 0
    while not state.terminated and (max_steps is None or step < max_steps):
        logits, value = model(state.observation[None, ...], use_running_average=True)
        logits = logits[0]
        # Mask invalid actions
        logits = jnp.where(
            state.legal_action_mask, logits, jnp.finfo(logits.dtype).min
        )
        action = int(jnp.argmax(logits))
        print(
            f"Step {step}: current_player={state.current_player}, action={action}, value={float(value[0]):.3f}"
        )
        state = env.step(state, action)
        step += 1
    if state.terminated:
        print(f"\nGame finished. Rewards: {state.rewards}")
    else:
        print(f"\nStopped after {max_steps} steps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()
    run_model(args.checkpoint_dir)
