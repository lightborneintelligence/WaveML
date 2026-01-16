"""
WaveML-JAX: Wave-Native Machine Learning
========================================
Lightborne Intelligence

Error Rectification by Alignment (ERA) - Reference Implementation

A pure functional implementation of wave-native computing in JAX.

Usage:
    from waveml_jax.core import WaveState, era_rectify, DEFAULT_BOUNDS
    from waveml_jax.models import WaveSeq, WaveSeqParams
    from waveml_jax.benchmarks import run_benchmark

The wave IS the computation.
"""

__version__ = "1.0.0"
__author__ = "Lightborne Intelligence"
__license__ = "Apache-2.0"

from .core import (
    WaveState,
    era_rectify,
    DEFAULT_BOUNDS,
    TIGHT_BOUNDS,
    LOOSE_BOUNDS,
    InvariantBounds,
)

__all__ = [
    "WaveState",
    "era_rectify",
    "DEFAULT_BOUNDS",
    "TIGHT_BOUNDS",
    "LOOSE_BOUNDS",
    "InvariantBounds",
    "__version__",
]
