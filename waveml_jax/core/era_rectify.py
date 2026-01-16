# “””
WaveML-JAX: Error Rectification by Alignment (ERA)

Lightborne Intelligence

ERA enforces physical invariants at every computational step.
This is NOT a post-hoc correction but an integral part of state evolution.

The rectification operator:
Ψ_{t+1} ← R(Ψ_t)

where R enforces invariant compliance before further propagation.

Key insight: ERA does not damp or suppress states arbitrarily.
It realigns them to the nearest admissible configuration,
preserving information content while ensuring physical validity.
“””

import jax
import jax.numpy as jnp
from typing import Tuple, Optional
from functools import partial

try:
from .representation import WaveState, total_energy
except ImportError:
from representation import WaveState, total_energy

try:
from .invariants import InvariantBounds, DEFAULT_BOUNDS, OMEGA
except ImportError:
from invariants import InvariantBounds, DEFAULT_BOUNDS, OMEGA

# ============================================================================

# Core ERA Rectification

# ============================================================================

@jax.jit
def era_rectify(state: WaveState,
bounds: InvariantBounds = DEFAULT_BOUNDS) -> WaveState:
“””
Enforce wave invariants through rectification.

```
This is the core ERA operator. It enforces:
    1. Amplitude non-negativity
    2. Amplitude upper bound (element-wise)
    3. Total energy bound (global) - THE KEY CONSTRAINT
    4. Phase wrapping to [-π, π]
    5. Phase gating where amplitude vanishes

Args:
    state: Wave state to rectify
    bounds: Invariant bounds to enforce

Returns:
    Rectified wave state satisfying all invariants
"""
amp = state.amplitude
phase = state.phase

# 1. Amplitude non-negativity
amp = jnp.maximum(amp, bounds.min_amplitude)

# 2. Amplitude upper bound (element-wise)
amp = jnp.minimum(amp, bounds.max_amplitude)

# 3. Total energy bound (THE KEY)
# This is what prevents runaway dynamics
energy = jnp.sum(amp ** 2, axis=-1, keepdims=True)
scale = jnp.where(
    energy > bounds.max_energy,
    jnp.sqrt(bounds.max_energy / (energy + OMEGA)),
    1.0
)
amp = amp * scale

# 4. Phase wrapping (gradient-safe via atan2)
# This avoids discontinuities in the gradient
phase = jnp.arctan2(jnp.sin(phase), jnp.cos(phase))

# 5. Phase gating - freeze gradients where amplitude vanishes
# This prevents phantom phase gradients from corrupting training
safe = amp > bounds.phase_gate_threshold
phase = jnp.where(safe, phase, jax.lax.stop_gradient(phase))

return WaveState(amplitude=amp, phase=phase)
```

@jax.jit
def era_rectify_soft(state: WaveState,
bounds: InvariantBounds = DEFAULT_BOUNDS,
temperature: float = 1.0) -> WaveState:
“””
Soft ERA rectification using smooth approximations.

```
Better gradient flow but less strict enforcement.
Use for training; switch to hard rectification for inference.

Args:
    state: Wave state to rectify
    bounds: Invariant bounds to enforce
    temperature: Softness parameter (lower = harder)
"""
amp = state.amplitude
phase = state.phase

# Soft non-negativity via softplus
amp = jax.nn.softplus(amp * temperature) / temperature

# Soft upper bound via sigmoid scaling
amp = bounds.max_amplitude * jax.nn.sigmoid(
    (amp - bounds.max_amplitude / 2) / (bounds.max_amplitude * temperature)
) * 2

# Energy scaling (same as hard version)
energy = jnp.sum(amp ** 2, axis=-1, keepdims=True)
scale = jnp.where(
    energy > bounds.max_energy,
    jnp.sqrt(bounds.max_energy / (energy + OMEGA)),
    1.0
)
amp = amp * scale

# Phase wrapping (same)
phase = jnp.arctan2(jnp.sin(phase), jnp.cos(phase))

# Soft phase gating via sigmoid
gate = jax.nn.sigmoid((amp - bounds.phase_gate_threshold) / OMEGA)
phase = gate * phase + (1 - gate) * jax.lax.stop_gradient(phase)

return WaveState(amplitude=amp, phase=phase)
```

# ============================================================================

# Component-wise Rectification (for debugging/analysis)

# ============================================================================

@jax.jit
def rectify_amplitude(amp: jnp.ndarray,
bounds: InvariantBounds = DEFAULT_BOUNDS) -> jnp.ndarray:
“”“Rectify amplitude only (non-negativity + upper bound).”””
amp = jnp.maximum(amp, bounds.min_amplitude)
amp = jnp.minimum(amp, bounds.max_amplitude)
return amp

@jax.jit
def rectify_energy(amp: jnp.ndarray,
bounds: InvariantBounds = DEFAULT_BOUNDS) -> jnp.ndarray:
“”“Rectify amplitude to satisfy energy bound.”””
energy = jnp.sum(amp ** 2, axis=-1, keepdims=True)
scale = jnp.where(
energy > bounds.max_energy,
jnp.sqrt(bounds.max_energy / (energy + OMEGA)),
1.0
)
return amp * scale

@jax.jit
def rectify_phase(phase: jnp.ndarray) -> jnp.ndarray:
“”“Wrap phase to [-π, π].”””
return jnp.arctan2(jnp.sin(phase), jnp.cos(phase))

# ============================================================================

# ERA with Diagnostics

# ============================================================================

@jax.jit
def era_rectify_with_stats(state: WaveState,
bounds: InvariantBounds = DEFAULT_BOUNDS) -> Tuple[WaveState, dict]:
“””
ERA rectification with diagnostic statistics.

```
Returns:
    Tuple of (rectified_state, stats_dict)
"""
amp_before = state.amplitude
phase_before = state.phase
energy_before = jnp.sum(amp_before ** 2, axis=-1)

# Apply rectification
rectified = era_rectify(state, bounds)

amp_after = rectified.amplitude
phase_after = rectified.phase
energy_after = jnp.sum(amp_after ** 2, axis=-1)

stats = {
    'amplitude_change': jnp.mean(jnp.abs(amp_after - amp_before)),
    'phase_change': jnp.mean(jnp.abs(phase_after - phase_before)),
    'energy_before': jnp.mean(energy_before),
    'energy_after': jnp.mean(energy_after),
    'energy_reduction': jnp.mean(energy_before - energy_after),
    'modes_clipped': jnp.mean(amp_before > bounds.max_amplitude),
    'modes_gated': jnp.mean(amp_after < bounds.phase_gate_threshold),
}

return rectified, stats
```

# ============================================================================

# Chained ERA (for sequence models)

# ============================================================================

def era_chain(states: list, bounds: InvariantBounds = DEFAULT_BOUNDS) -> list:
“”“Apply ERA to a sequence of states.”””
return [era_rectify(s, bounds) for s in states]

@partial(jax.jit, static_argnums=(2,))
def era_scan(initial_state: WaveState,
inputs: jnp.ndarray,
step_fn,
bounds: InvariantBounds = DEFAULT_BOUNDS) -> Tuple[WaveState, jnp.ndarray]:
“””
Scan with ERA applied after each step.

```
This is the pattern for sequence models:
    state_{t+1} = ERA(step_fn(state_t, input_t))

Args:
    initial_state: Starting wave state
    inputs: Input sequence, shape (seq_len, ...)
    step_fn: Function (state, input) -> state
    bounds: ERA bounds

Returns:
    Tuple of (final_state, all_states_stacked)
"""
def scan_body(state, x):
    new_state = step_fn(state, x)
    rectified = era_rectify(new_state, bounds)
    return rectified, rectified

final, trajectory = jax.lax.scan(scan_body, initial_state, inputs)
return final, trajectory
```

# ============================================================================

# Tests

# ============================================================================

def test_era_rectify():
“”“Test ERA rectification.”””
print(”=” * 60)
print(”  ERA Rectification Tests”)
print(”=” * 60)

```
try:
    from .representation import WaveState, total_energy, random
    from .invariants import is_stable, measure_stability
except ImportError:
    from representation import WaveState, total_energy, random
    from invariants import is_stable, measure_stability

bounds = DEFAULT_BOUNDS

# Test 1: Valid state passes through unchanged
print("\n[1] Valid state passthrough...")
valid = WaveState(
    amplitude=jnp.array([1.0, 2.0, 1.5]),
    phase=jnp.array([0.5, -0.3, 1.0])
)
rectified = era_rectify(valid, bounds)
amp_diff = jnp.max(jnp.abs(rectified.amplitude - valid.amplitude))
phase_diff = jnp.max(jnp.abs(rectified.phase - valid.phase))
print(f"    ✓ Amplitude change: {amp_diff:.6f}")
print(f"    ✓ Phase change: {phase_diff:.6f}")
assert amp_diff < 1e-6, "Valid amplitude should be unchanged"
assert phase_diff < 1e-6, "Valid phase should be unchanged"

# Test 2: Negative amplitude rectification
print("\n[2] Negative amplitude rectification...")
negative = WaveState(
    amplitude=jnp.array([-1.0, 2.0, -0.5]),
    phase=jnp.array([0.0, 0.0, 0.0])
)
rectified = era_rectify(negative, bounds)
print(f"    Before: {negative.amplitude}")
print(f"    After:  {rectified.amplitude}")
assert jnp.all(rectified.amplitude >= 0), "Should be non-negative"

# Test 3: Amplitude clipping
print("\n[3] Amplitude clipping...")
too_high = WaveState(
    amplitude=jnp.array([1.0, 15.0, 1.0]),  # 15 > max=10
    phase=jnp.array([0.0, 0.0, 0.0])
)
rectified = era_rectify(too_high, bounds)
print(f"    Before: {too_high.amplitude}")
print(f"    After:  {rectified.amplitude}")
assert jnp.all(rectified.amplitude <= bounds.max_amplitude), "Should be clipped"

# Test 4: Energy scaling (THE KEY TEST)
print("\n[4] Energy scaling...")
high_energy = WaveState(
    amplitude=jnp.ones(50) * 3.0,  # Energy = 9 * 50 = 450 > 100
    phase=jnp.zeros(50)
)
energy_before = total_energy(high_energy)
rectified = era_rectify(high_energy, bounds)
energy_after = total_energy(rectified)
print(f"    Energy before: {energy_before:.1f}")
print(f"    Energy after:  {energy_after:.1f}")
print(f"    Max allowed:   {bounds.max_energy:.1f}")
assert energy_after <= bounds.max_energy + OMEGA, "Energy should be bounded"

# Test 5: Phase wrapping
print("\n[5] Phase wrapping...")
unwrapped = WaveState(
    amplitude=jnp.array([1.0, 1.0, 1.0]),
    phase=jnp.array([5.0, -5.0, 10.0])  # Outside [-π, π]
)
rectified = era_rectify(unwrapped, bounds)
print(f"    Before: {unwrapped.phase}")
print(f"    After:  {rectified.phase}")
assert jnp.all(jnp.abs(rectified.phase) <= jnp.pi + 1e-6), "Phase should be wrapped"

# Test 6: Stability after rectification
print("\n[6] Stability guarantee...")
key = jax.random.PRNGKey(42)
for i in range(5):
    k = jax.random.fold_in(key, i)
    # Create deliberately unstable state
    wild = WaveState(
        amplitude=jax.random.uniform(k, (32,)) * 20 - 5,  # Can be negative or >10
        phase=jax.random.uniform(k, (32,)) * 20 - 10  # Way outside [-π, π]
    )
    rectified = era_rectify(wild, bounds)
    stable = is_stable(rectified, bounds)
    assert stable, f"Rectified state should always be stable (trial {i})"
print(f"    ✓ All 5 random trials stable after rectification")

# Test 7: ERA with diagnostics
print("\n[7] ERA with diagnostics...")
high_energy = WaveState(
    amplitude=jnp.ones(20) * 5.0,
    phase=jnp.linspace(0, 4 * jnp.pi, 20)
)
rectified, stats = era_rectify_with_stats(high_energy, bounds)
print(f"    Energy before: {stats['energy_before']:.1f}")
print(f"    Energy after:  {stats['energy_after']:.1f}")
print(f"    Energy reduction: {stats['energy_reduction']:.1f}")
print(f"    Amplitude change: {stats['amplitude_change']:.3f}")
print(f"    Phase change: {stats['phase_change']:.3f}")

# Test 8: Differentiability
print("\n[8] Differentiability...")
def loss_fn(amp):
    state = WaveState(amplitude=amp, phase=jnp.zeros_like(amp))
    rect = era_rectify(state, bounds)
    return jnp.sum(rect.amplitude ** 2)

amp = jnp.array([5.0, 10.0, 15.0])
grad = jax.grad(loss_fn)(amp)
print(f"    Input amplitude: {amp}")
print(f"    Gradient: {grad}")
assert not jnp.any(jnp.isnan(grad)), "Gradients should not be NaN"

# Test 9: Batched operation
print("\n[9] Batched operation...")
batch_state = WaveState(
    amplitude=jnp.ones((4, 8, 16)) * 3.0,
    phase=jnp.zeros((4, 8, 16))
)
rectified = era_rectify(batch_state, bounds)
print(f"    Input shape: {batch_state.amplitude.shape}")
print(f"    Output shape: {rectified.amplitude.shape}")
assert rectified.amplitude.shape == batch_state.amplitude.shape

# Test 10: Soft vs hard rectification
print("\n[10] Soft vs hard comparison...")
state = WaveState(
    amplitude=jnp.array([8.0, 12.0, 3.0]),
    phase=jnp.array([0.0, 0.0, 0.0])
)
hard = era_rectify(state, bounds)
soft = era_rectify_soft(state, bounds, temperature=0.1)
print(f"    Original:  {state.amplitude}")
print(f"    Hard ERA:  {hard.amplitude}")
print(f"    Soft ERA:  {soft.amplitude}")

print("\n" + "=" * 60)
print("  All ERA tests passed! ✓")
print("=" * 60)

return True
```

if **name** == “**main**”:
test_era_rectify()
