"""Data generation, serialization, and grouped split utilities."""

from .real import RealBundle, build_real_cache, validate_real_cache
from .real_panel import RealPanel, generate_real_panel
from .synthetic import SyntheticBundle, generate_synthetic_bundle

__all__ = [
    "RealBundle",
    "RealPanel",
    "SyntheticBundle",
    "build_real_cache",
    "generate_real_panel",
    "generate_synthetic_bundle",
    "validate_real_cache",
]
