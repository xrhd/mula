import unittest

import jax
import jax.numpy as jnp
from flax import nnx

from mula.models.alphazero.modeling import AZNet, BlockV1, BlockV2


class TestBlockV1(unittest.TestCase):
    def test_output_shape(self):
        rngs = nnx.Rngs(0)
        block = BlockV1(num_channels=64, rngs=rngs)
        x = jnp.ones((2, 8, 8, 64))
        y = block(x, use_running_average=False)
        self.assertEqual(y.shape, (2, 8, 8, 64))

    def test_eval_mode(self):
        rngs = nnx.Rngs(0)
        block = BlockV1(num_channels=64, rngs=rngs)
        x = jnp.ones((2, 8, 8, 64))
        y = block(x, use_running_average=True)
        self.assertEqual(y.shape, (2, 8, 8, 64))


class TestBlockV2(unittest.TestCase):
    def test_output_shape(self):
        rngs = nnx.Rngs(0)
        block = BlockV2(num_channels=64, rngs=rngs)
        x = jnp.ones((2, 8, 8, 64))
        y = block(x, use_running_average=False)
        self.assertEqual(y.shape, (2, 8, 8, 64))

    def test_eval_mode(self):
        rngs = nnx.Rngs(0)
        block = BlockV2(num_channels=64, rngs=rngs)
        x = jnp.ones((2, 8, 8, 64))
        y = block(x, use_running_average=True)
        self.assertEqual(y.shape, (2, 8, 8, 64))


class TestAZNet(unittest.TestCase):
    def test_forward_shape(self):
        rngs = nnx.Rngs(0)
        model = AZNet(num_actions=65, num_channels=64, num_blocks=3, resnet_v2=True, rngs=rngs)
        x = jnp.ones((2, 8, 8, 2))
        logits, value = model(x, use_running_average=False)
        self.assertEqual(logits.shape, (2, 65))
        self.assertEqual(value.shape, (2,))

    def test_forward_shape_single(self):
        rngs = nnx.Rngs(0)
        model = AZNet(num_actions=65, num_channels=64, num_blocks=3, resnet_v2=True, rngs=rngs)
        x = jnp.ones((1, 8, 8, 2))
        logits, value = model(x, use_running_average=False)
        self.assertEqual(logits.shape, (1, 65))
        self.assertEqual(value.shape, (1,))

    def test_eval_mode(self):
        rngs = nnx.Rngs(0)
        model = AZNet(num_actions=65, num_channels=64, num_blocks=3, resnet_v2=True, rngs=rngs)
        x = jnp.ones((2, 8, 8, 2))
        logits, value = model(x, use_running_average=True)
        self.assertEqual(logits.shape, (2, 65))
        self.assertEqual(value.shape, (2,))

    def test_train_mode_updates_bn(self):
        rngs = nnx.Rngs(0)
        model = AZNet(num_actions=65, num_channels=64, num_blocks=3, resnet_v2=True, rngs=rngs)
        x = jnp.ones((2, 8, 8, 2))
        # Get initial batch stats
        graphdef, state = nnx.split(model)
        mean_before = state["final_bn"]["mean"].value.copy()
        model = nnx.merge(graphdef, state)
        _ = model(x, use_running_average=False)
        graphdef, state_after = nnx.split(model)
        mean_after = state_after["final_bn"]["mean"].value
        # Stats should have updated
        self.assertFalse(jnp.allclose(mean_before, mean_after))

    def test_eval_mode_does_not_update_bn(self):
        rngs = nnx.Rngs(0)
        model = AZNet(num_actions=65, num_channels=64, num_blocks=3, resnet_v2=True, rngs=rngs)
        x = jnp.ones((2, 8, 8, 2))
        graphdef, state = nnx.split(model)
        mean_before = state["final_bn"]["mean"].value.copy()
        model = nnx.merge(graphdef, state)
        _ = model(x, use_running_average=True)
        graphdef, state_after = nnx.split(model)
        mean_after = state_after["final_bn"]["mean"].value
        # Stats should NOT have updated
        self.assertTrue(jnp.allclose(mean_before, mean_after))

    def test_resnet_v1(self):
        rngs = nnx.Rngs(0)
        model = AZNet(num_actions=65, num_channels=64, num_blocks=3, resnet_v2=False, rngs=rngs)
        x = jnp.ones((2, 8, 8, 2))
        logits, value = model(x, use_running_average=False)
        self.assertEqual(logits.shape, (2, 65))
        self.assertEqual(value.shape, (2,))

    def test_value_range(self):
        rngs = nnx.Rngs(0)
        model = AZNet(num_actions=65, num_channels=64, num_blocks=3, resnet_v2=True, rngs=rngs)
        x = jax.random.normal(jax.random.PRNGKey(1), (4, 8, 8, 2))
        _, value = model(x, use_running_average=True)
        self.assertTrue(jnp.all(value >= -1.0))
        self.assertTrue(jnp.all(value <= 1.0))


if __name__ == "__main__":
    unittest.main()
