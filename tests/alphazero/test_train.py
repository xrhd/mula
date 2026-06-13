import os
import pickle
import tempfile
import unittest

import jax
import jax.numpy as jnp
import mctx
import optax
import pgx
from flax import nnx

from mula.models.alphazero.modeling import AZNet
from mula.models.alphazero.params import ModelConfig
from mula.models.alphazero.train import (
    Config,
    SelfplayOutput,
    Sample,
    baseline,
    _compute_loss_input_impl,
    _elo_from_win_rate,
    _ensure_config,
    _selfplay_impl,
    evaluate,
    loss_fn,
    make_model,
    recurrent_fn,
)


class TestRecurrentFn(unittest.TestCase):
    def test_recurrent_fn_output(self):
        _ensure_config()
        env = pgx.make("othello")
        config = Config()
        rngs = nnx.Rngs(0)
        model = make_model(rngs)
        state = env.init(jax.random.PRNGKey(0))
        # Batch the state properly for vmap
        def add_batch(x):
            if hasattr(x, 'shape') and len(x.shape) > 0:
                return x[None, ...]
            # scalars get a batch dim
            return jnp.asarray(x)[None, ...]
        states = jax.tree_util.tree_map(add_batch, state)
        action = jnp.zeros((1,), dtype=jnp.int32)
        graphdef, state = nnx.split(model)
        params_state = (graphdef, state)
        output, next_state = recurrent_fn(
            params_state, jax.random.PRNGKey(0), action, states
        )
        self.assertIsInstance(output, mctx.RecurrentFnOutput)
        self.assertEqual(output.reward.shape, (1,))
        self.assertEqual(output.discount.shape, (1,))
        self.assertEqual(output.prior_logits.shape, (1, env.num_actions))
        self.assertEqual(output.value.shape, (1,))


class TestSelfplay(unittest.TestCase):
    def test_selfplay_output_shape(self):
        _ensure_config()
        config = Config()
        rngs = nnx.Rngs(0)
        model = make_model(rngs)
        graphdef, state = nnx.split(model)
        params_state = (graphdef, state)
        rng_key = jax.random.PRNGKey(0)
        data = _selfplay_impl(params_state, rng_key)
        # Verify shapes are consistent with config
        self.assertEqual(data.obs.ndim, 5)
        self.assertEqual(data.action_weights.ndim, 3)
        self.assertEqual(data.reward.ndim, 2)
        self.assertEqual(data.terminated.ndim, 2)
        self.assertEqual(data.discount.ndim, 2)
        self.assertEqual(data.obs.shape[0], config.max_num_steps)
        self.assertEqual(data.obs.shape[1], config.selfplay_batch_size)
        self.assertEqual(data.obs.shape[2:], (8, 8, 2))
        self.assertEqual(data.action_weights.shape[2], 65)


class TestComputeLossInput(unittest.TestCase):
    def test_sample_shape(self):
        _ensure_config()
        config = Config()
        batch_size = config.selfplay_batch_size
        data = SelfplayOutput(
            obs=jnp.ones((config.max_num_steps, batch_size, 8, 8, 2)),
            reward=jnp.zeros((config.max_num_steps, batch_size)),
            terminated=jnp.zeros((config.max_num_steps, batch_size), dtype=jnp.bool_),
            action_weights=jnp.ones((config.max_num_steps, batch_size, 65)) / 65,
            discount=jnp.ones((config.max_num_steps, batch_size)) * -1.0,
        )
        sample = _compute_loss_input_impl(data)
        self.assertEqual(sample.obs.shape, data.obs.shape)
        self.assertEqual(sample.policy_tgt.shape, data.action_weights.shape)
        self.assertEqual(sample.value_tgt.shape, data.reward.shape)
        self.assertEqual(sample.mask.shape, data.terminated.shape)
        # No termination means no value targets (mask should be False)
        self.assertFalse(jnp.all(sample.mask))

    def test_value_target_computation(self):
        _ensure_config()
        config = Config()
        batch_size = config.selfplay_batch_size
        # Simple reward at last step
        reward = jnp.zeros((config.max_num_steps, batch_size))
        reward = reward.at[-1, :].set(1.0)
        terminated = jnp.zeros((config.max_num_steps, batch_size), dtype=jnp.bool_)
        terminated = terminated.at[-1, :].set(True)
        data = SelfplayOutput(
            obs=jnp.ones((config.max_num_steps, batch_size, 8, 8, 2)),
            reward=reward,
            terminated=terminated,
            action_weights=jnp.ones((config.max_num_steps, batch_size, 65)) / 65,
            discount=jnp.ones((config.max_num_steps, batch_size)) * -1.0,
        )
        sample = _compute_loss_input_impl(data)
        # Value target should be gamma-discounted back from last reward
        # For a single +1 reward at final step with gamma=-1, targets alternate
        self.assertTrue(jnp.allclose(sample.value_tgt[-1, :], 1.0))


class TestLossFn(unittest.TestCase):
    def test_loss_computation(self):
        rngs = nnx.Rngs(0)
        model = make_model(rngs)
        batch_size = 4
        samples = Sample(
            obs=jnp.ones((batch_size, 8, 8, 2)),
            policy_tgt=jnp.ones((batch_size, 65)) / 65,
            value_tgt=jnp.zeros((batch_size,)),
            mask=jnp.ones((batch_size,), dtype=jnp.bool_),
        )
        loss, (policy_loss, value_loss) = loss_fn(model, samples)
        self.assertGreater(loss, 0)
        self.assertGreaterEqual(policy_loss, 0)
        self.assertGreaterEqual(value_loss, 0)


class TestTrainStep(unittest.TestCase):
    def test_train_step_updates(self):
        # Use a small model for speed
        rngs = nnx.Rngs(0)
        model = AZNet(num_actions=65, num_channels=32, num_blocks=2, resnet_v2=True, rngs=rngs)
        optimizer = nnx.Optimizer(model, optax.adam(learning_rate=0.001), wrt=nnx.Param)
        batch_size = 4
        samples = Sample(
            obs=jnp.ones((batch_size, 8, 8, 2)),
            policy_tgt=jnp.ones((batch_size, 65)) / 65,
            value_tgt=jnp.zeros((batch_size,)),
            mask=jnp.ones((batch_size,), dtype=jnp.bool_),
        )
        # Test on single device without pmap
        grads, (policy_loss, value_loss) = nnx.grad(loss_fn, has_aux=True)(
            model, samples
        )
        optimizer.update(model, grads)
        self.assertGreater(policy_loss, 0)
        self.assertGreater(value_loss, 0)
        # Verify model is still valid by running forward pass
        logits, value = model(samples.obs, use_running_average=False)
        self.assertEqual(logits.shape, (batch_size, 65))
        self.assertEqual(value.shape, (batch_size,))


class TestEvaluate(unittest.TestCase):
    def test_evaluate_shape(self):
        # Skip if baseline not available
        _ensure_config()
        if baseline is None:
            self.skipTest("Baseline model not available")
        rngs = nnx.Rngs(0)
        model = make_model(rngs)
        rng_key = jax.random.PRNGKey(0)
        # evaluate is pmapped; test with a single device batch
        R = evaluate(rng_key, model)
        self.assertEqual(R.shape, (Config().selfplay_batch_size,))
        self.assertTrue(jnp.all((R >= -1) & (R <= 1)))


class TestEloFromWinRate(unittest.TestCase):
    def test_perfect_win(self):
        self.assertAlmostEqual(_elo_from_win_rate(1.0), 400, places=5)

    def test_perfect_loss(self):
        self.assertAlmostEqual(_elo_from_win_rate(0.0), -400, places=5)

    def test_even(self):
        self.assertAlmostEqual(_elo_from_win_rate(0.5), 0, places=5)


class TestCheckpointing(unittest.TestCase):
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rngs = nnx.Rngs(0)
            model = make_model(rngs)
            config = Config()
            rng_key = jax.random.PRNGKey(0)
            path = os.path.join(tmpdir, "test.ckpt")
            with open(path, "wb") as f:
                dic = {
                    "config": config,
                    "rng_key": rng_key,
                    "model": model,
                    "iteration": 10,
                }
                pickle.dump(dic, f)
            with open(path, "rb") as f:
                loaded = pickle.load(f)
            self.assertEqual(loaded["iteration"], 10)
            self.assertEqual(loaded["config"].env_id, "othello")


if __name__ == "__main__":
    unittest.main()
