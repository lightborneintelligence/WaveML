"""
WaveML Core Module
==================
Lightborne Intelligence

Core components for wave-native computing:
- representation: WaveState, encoding/decoding
- invariants: Bounds and stability metrics
- era_rectify: Error Rectification by Alignment
"""

from .representation import (
    WaveState,
    to_complex,
    from_complex,
    to_real,
    from_real,
    encode_fft,
    decode_fft,
    energy,
    total_energy,
    phase_coherence,
    zeros,
    ones,
    random,
)

from .invariants import (
    OMEGA,
    InvariantBounds,
    DEFAULT_BOUNDS,
    TIGHT_BOUNDS,
    LOOSE_BOUNDS,
    measure_stability,
    is_stable,
    amplitude_violation_loss,
    energy_violation_loss,
    invariant_loss,
    scale_bounds,
    bounds_for_n_modes,
)

from .era_rectify import (
    era_rectify,
    era_rectify_soft,
    era_rectify_with_stats,
    rectify_amplitude,
    rectify_energy,
    rectify_phase,
    era_chain,
    era_scan,
)

__all__ = [
    "WaveState",
    "to_complex",
    "from_complex",
    "to_real",
    "from_real",
    "encode_fft",
    "decode_fft",
    "energy",
    "total_energy",
    "phase_coherence",
    "zeros",
    "ones",
    "random",
    "OMEGA",
    "InvariantBounds",
    "DEFAULT_BOUNDS",
    "TIGHT_BOUNDS",
    "LOOSE_BOUNDS",
    "measure_stability",
    "is_stable",
    "amplitude_violation_loss",
    "energy_violation_loss",
    "invariant_loss",
    "scale_bounds",
    "bounds_for_n_modes",
    "era_rectify",
    "era_rectify_soft",
    "era_rectify_with_stats",
    "rectify_amplitude",
    "rectify_energy",
    "rectify_phase",
    "era_chain",
    "era_scan",
]
