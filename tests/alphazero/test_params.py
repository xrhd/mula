import dataclasses
import os
import unittest

from mula.models.alphazero.params import ModelConfig


class TestModelConfig(unittest.TestCase):
    def test_default_values(self):
        cfg = ModelConfig()
        self.assertEqual(cfg.env_id, "othello")
        self.assertEqual(cfg.seed, 0)
        self.assertEqual(cfg.max_num_iters, 400)
        self.assertEqual(cfg.num_channels, 128)
        self.assertEqual(cfg.num_layers, 6)
        self.assertTrue(cfg.resnet_v2)
        self.assertEqual(cfg.selfplay_batch_size, 1024)
        self.assertEqual(cfg.num_simulations, 32)
        self.assertEqual(cfg.max_num_steps, 256)
        self.assertEqual(cfg.training_batch_size, 4096)
        self.assertEqual(cfg.learning_rate, 0.001)
        self.assertEqual(cfg.eval_interval, 5)
        self.assertEqual(cfg.checkpoint_dir, "checkpoints")
        self.assertEqual(cfg.plot_dir, "plots")

    def test_custom_values(self):
        cfg = ModelConfig(num_channels=64, num_layers=4, learning_rate=0.01)
        self.assertEqual(cfg.num_channels, 64)
        self.assertEqual(cfg.num_layers, 4)
        self.assertEqual(cfg.learning_rate, 0.01)
        # Defaults remain
        self.assertEqual(cfg.env_id, "othello")

    def test_invalid_num_channels(self):
        with self.assertRaises(ValueError):
            ModelConfig(num_channels=0)

    def test_invalid_num_layers(self):
        with self.assertRaises(ValueError):
            ModelConfig(num_layers=-1)

    def test_invalid_selfplay_batch_size(self):
        with self.assertRaises(ValueError):
            ModelConfig(selfplay_batch_size=0)

    def test_invalid_num_simulations(self):
        with self.assertRaises(ValueError):
            ModelConfig(num_simulations=0)

    def test_invalid_max_num_steps(self):
        with self.assertRaises(ValueError):
            ModelConfig(max_num_steps=0)

    def test_invalid_training_batch_size(self):
        with self.assertRaises(ValueError):
            ModelConfig(training_batch_size=-1)

    def test_invalid_learning_rate(self):
        with self.assertRaises(ValueError):
            ModelConfig(learning_rate=0)

    def test_frozen(self):
        cfg = ModelConfig()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cfg.seed = 1


if __name__ == "__main__":
    unittest.main()
