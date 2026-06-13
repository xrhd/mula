import os
import pickle
import tempfile
import unittest

import jax
from flax import nnx

from mula.models.alphazero.modeling import AZNet
from mula.models.alphazero.params import ModelConfig
from mula.models.alphazero.run_model import run_model


class TestRunModel(unittest.TestCase):
    def test_run_model_fresh(self):
        # Should run without error using a fresh model
        try:
            run_model(checkpoint_dir=None, max_steps=5)
        except Exception as e:
            self.fail(f"run_model raised an exception: {e}")

    def test_run_model_with_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake checkpoint with proper module format
            rngs = nnx.Rngs(0)
            model = AZNet(
                num_actions=65,
                num_channels=64,
                num_blocks=3,
                resnet_v2=True,
                rngs=rngs,
            )
            path = os.path.join(tmpdir, "000000.ckpt")
            with open(path, "wb") as f:
                pickle.dump({"model": model}, f)
            try:
                run_model(checkpoint_dir=tmpdir, max_steps=5)
            except Exception as e:
                self.fail(f"run_model with checkpoint raised an exception: {e}")

    def test_run_model_empty_checkpoint_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                run_model(checkpoint_dir=tmpdir, max_steps=5)
            except Exception as e:
                self.fail(f"run_model with empty dir raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
