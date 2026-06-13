import subprocess
import sys

# Ensure the parent package is importable
sys.path.insert(0, "/content")

# Install dependencies if any are missing
_MISSING_DEPS = []
for _pkg in ("pgx", "mctx", "matplotlib", "omegaconf", "pydantic", "flax", "jax"):
    try:
        __import__(_pkg)
    except ImportError:
        _MISSING_DEPS.append(_pkg)

if _MISSING_DEPS:
    print("Installing missing dependencies:", _MISSING_DEPS)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *_MISSING_DEPS])
    # Re-import to ensure they are available in this process
    for _pkg in _MISSING_DEPS:
        __import__(_pkg)

import runpy

runpy.run_module("mula.models.alphazero.train", run_name="__main__")
