"""Backbone namespace for linear probing experiments.

Avoid importing heavyweight/optional dependencies at module import time.
"""

from .dino import DINOBackbone, DINOv3Backbone
from .dinov2 import DINOv2Backbone
from .linear_head import MultiLayerLinearHead

__all__ = [
    "DINOBackbone",
    "DINOv3Backbone",
    "DINOv2Backbone",
    "MultiLayerLinearHead",
]

try:
    from .openclip import OpenCLIPBackbone
except Exception:
    OpenCLIPBackbone = None
else:
    __all__.append("OpenCLIPBackbone")

try:
    from .sam import SAMBackbone
except Exception:
    SAMBackbone = None
else:
    __all__.append("SAMBackbone")

# Optional: SigLIP2 backbone (depends on transformers)
try:  # pragma: no cover - optional dependency
    from .siglip2 import SigLIP2Backbone  # type: ignore
except Exception:
    SigLIP2Backbone = None  # type: ignore
else:
    __all__.append("SigLIP2Backbone")

# Optional: Flux backbone (depends on transformers + flash_attn)
try:  # pragma: no cover - optional dependency
    from .flux import FluxBackbone  # type: ignore
except Exception:
    FluxBackbone = None  # type: ignore
else:
    __all__.append("FluxBackbone")

# Optional: Stable Diffusion backbone
try:  # pragma: no cover - optional dependency
    from .stable_diffusion import StableDiffusionBackbone  # type: ignore
except Exception:
    StableDiffusionBackbone = None  # type: ignore
else:
    __all__.append("StableDiffusionBackbone")

# Optional: SigLIP So400m backbone (raw, PaliGemma, pi0, pi0.5 sources)
try:  # pragma: no cover - optional dependency
    from .siglip_so400m import SigLIPSo400mBackbone  # type: ignore
except Exception:
    SigLIPSo400mBackbone = None  # type: ignore
else:
    __all__.append("SigLIPSo400mBackbone")
