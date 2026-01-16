# “””
WaveML-JAX: Wave-Native Machine Learning

Lightborne Intelligence

Error Rectification by Alignment (ERA) - Reference Implementation

A pure functional implementation of wave-native computing in JAX.

Usage:
from waveml_jax.core import WaveState, era_rectify, DEFAULT_BOUNDS
from waveml_jax.models import WaveSeq, WaveSeqParams
from waveml_jax.benchmarks import run_benchmark

The wave IS the computation.
“””

**version** = “1.0.0”
**author** = “Lightborne Intelligence”
**license** = “Apache-2.0”

# Convenient top-level imports

from .core import (
WaveState,
era_rectify,
DEFAULT_BOUNDS,
TIGHT_BOUNDS,
LOOSE_BOUNDS,
InvariantBounds,
)

**all** = [
‘WaveState’,
‘era_rectify’,
‘DEFAULT_BOUNDS’,
‘TIGHT_BOUNDS’,
‘LOOSE_BOUNDS’,
‘InvariantBounds’,
‘**version**’,
]