# AlphaZero for Othello

A clean Flax NNX implementation of (Gumbel) AlphaZero for Othello using [pgx](https://github.com/sotetsuk/pgx) and [mctx](https://github.com/deepmind/mctx).

## Files

- `modeling.py` — Flax NNX `AZNet` with ResNet blocks and policy/value heads.
- `params.py` — `ModelConfig` dataclass with Othello defaults.
- `train.py` — Self-play loop, MCTS, training, checkpointing, and plotting.
- `run_model.py` — Run a trained model to play Othello.

## Usage

### Training locally

```bash
python -m mula.models.alphazero.train env_id=othello seed=0 max_num_iters=10
```

### Training on remote Colab

```bash
make colab-new-gpu
make colab-upload
make colab-run-exec
make colab-download
make colab-stop
```

### Running the model

```bash
python -m mula run --model-name alphazero --path-root /tmp/checkpoints
```

Or directly:

```bash
python -m mula.models.alphazero.run_model --checkpoint-dir checkpoints/othello_xxx
```

## Outputs

- Checkpoints are saved under `checkpoints/othello_YYYYMMDDhhmmss/`.
- Training plots and an animated GIF are saved under `plots/othello_YYYYMMDDhhmmss/`.
