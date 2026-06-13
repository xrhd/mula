from mula.models.alphazero.modeling import AZNet
from mula.models.alphazero.params import ModelConfig

__all__ = ["AZNet", "ModelConfig"]

# Lazy import to avoid pulling in heavy deps (pgx, etc.) at package load time.
def __getattr__(name: str):
    if name == "run_model":
        from mula.models.alphazero.run_model import run_model as _run_model

        return _run_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
