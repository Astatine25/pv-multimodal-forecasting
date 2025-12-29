from .cnn_encoder import CNNEncoder
from .multimodal_transformer import MultimodalTransformer
from .vit_encoder import ViTEncoder
from .gnn_model import PVGraphModel
# Re-export the MultimodalTransformer class from the correct module.
# Try the common module names and raise a clear error if none found.

_try_imported = False
for _name in ("multimodal_transformer", "transformer"):
    try:
        module = __import__(f"{__name__}.{_name}", fromlist=["MultimodalTransformer"])
        MultimodalTransformer = getattr(module, "MultimodalTransformer")
        _try_imported = True
        break
    except (ImportError, AttributeError):
        continue

if not _try_imported:
    raise ImportError(
        "Could not import 'MultimodalTransformer' from models.multimodal_transformer or models.transformer. "
        "Check module filenames inside the models/ directory and adjust models/__init__.py accordingly."
    )

__all__ = ["MultimodalTransformer"]
