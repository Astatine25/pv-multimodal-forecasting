"""
inference package initializer.

This file tries to locate and re-export a `Predictor` class from a few
common submodule names so users can do:

    from inference import Predictor

If no concrete Predictor implementation is found, a helpful ImportError is
raised when someone tries to instantiate `Predictor`, allowing imports of the
package without failing immediately.
"""

__version__ = "0.0.0"

# Try importing Predictor from likely submodules. Adjust the list if your
# project uses a different module name.
_predictor_impl = None
for _mod in ("predict", "predictor", "runner", "core"):
    try:
        _module = __import__(f"{__name__}.{_mod}", fromlist=["Predictor"])
        _predictor_impl = getattr(_module, "Predictor")
        break
    except Exception:
        # Ignore and try the next candidate module name
        _predictor_impl = None

if _predictor_impl is None:
    class _MissingPredictor:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "No `Predictor` implementation found in the `inference` package. "
                "Expected `Predictor` to be defined in one of: "
                "inference.predict, inference.predictor, inference.runner, or inference.core. "
                "Either implement `Predictor` in a submodule or import the concrete "
                "module directly (for example `from inference.predict import Predictor`)."
            )

    Predictor = _MissingPredictor
else:
    Predictor = _predictor_impl

__all__ = ["Predictor"]
