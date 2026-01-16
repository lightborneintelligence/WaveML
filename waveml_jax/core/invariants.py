# “””
WaveML-JAX: Invariant Bounds

Lightborne Intelligence

Defines the invariant bounds that ERA enforces.
These bounds define the space of valid wave trajectories.

Core Invariants:
1. Amplitude non-negativity (energy cannot be negative)
2. Amplitude upper bound (no single mode dominates)
3. Total energy bound (global energy budget)
4. Phase wrapping (confined to [-π, π])
5. Phase gating (gradients frozen where amplitude vanishes)
“””

import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional
from functools import partial

try:
from .representation import WaveState, total_energy
except ImportError:
from representation import WaveState, total_energy

# ============================================================================

# Physical Constants

# ============================================================================

# Ω (Omega) - The stability threshold

# Named for the natural frequency of harmonic systems

OMEGA = 1e-6

# ============================================================================

# Invariant Bounds

# ============================================================================

class InvariantBounds(NamedTuple):
“””
Bounds defining the space of valid wave states.

```
Attributes:
    min_amplitude: Floor for amplitude (usually 0)
    max_amplitude: Ceiling for individual mode amplitude
    max_energy: Global energy budget (THE KEY CONSTRAINT)
    phase_gate_threshold: Below this amplitude, phase gradients frozen
"""
min_amplitude: float = 0.0
max_amplitude: float = 10.0
max_energy: float = 100.0
phase_gate_threshold: float = OMEGA
```

# Default bounds for general use

DEFAULT_BOUNDS = InvariantBounds()

# Tight bounds for stability-critical applications

TIGHT_BOUNDS = InvariantBounds(
max_amplitude=5.0,
max_energy=50.0
)

# Loose bounds for expressive applications

LOOSE_BOUNDS = InvariantBounds(
max_amplitude=20.0,
max_energy=400.0
)

# ============================================================================

# Stability Metrics

# ============================================================================

@jax.jit
def measure_stability(state: WaveState, bounds: InvariantBounds = DEFAULT_BOUNDS) -> dict:
“””
Measure how close a state is to violating invariants.

```
Returns dict with:
    - amplitude_margin: How far below max_amplitude (min across modes)
    - energy_margin: How far below max_energy
    - phase_wrapped: Whether all phases in [-π, π]
    - overall: Combined stability score in [0, 1]
"""
# Amplitude margin (per-mode, take minimum)
amp_margin = (bounds.max_amplitude - jnp.max(state.amplitude)) / bounds.max_amplitude
amp_margin = jnp.clip(amp_margin, 0.0, 1.0)

# Energy margin
e = total_energy(state)
energy_margin = (bounds.max_energy - e) / bounds.max_energy
energy_margin = jnp.clip(energy_margin, 0.0, 1.0)

# Phase check (should be in [-π, π])
phase_ok = jnp.all(jnp.abs(state.phase) <= jnp.pi + 1e-6)

# Overall stability (geometric mean)
overall = jnp.sqrt(amp_margin * energy_margin)

return {
    'amplitude_margin': amp_margin,
    'energy_margin': energy_margin,
    'phase_wrapped': phase_ok,
    'overall': overall
}
```

def is_stable(state: WaveState, bounds: InvariantBounds = DEFAULT_BOUNDS) -> bool:
“”“Check if state satisfies all invariants.”””
tol = 1e-4  # Tolerance for floating point comparisons

```
# Non-negativity
if jnp.any(state.amplitude < -tol):
    return False

# Amplitude bound
if jnp.any(state.amplitude > bounds.max_amplitude + tol):
    return False

# Energy bound
if total_energy(state) > bounds.max_energy + tol:
    return False

# Phase wrapping
if jnp.any(jnp.abs(state.phase) > jnp.pi + tol):
    return False

return True
```

# ============================================================================

# Invariant Violation Losses

# ============================================================================

@jax.jit
def amplitude_violation_loss(state: WaveState, bounds: InvariantBounds = DEFAULT_BOUNDS) -> float:
“”“Loss for amplitude bound violations.”””
# Negative amplitude violation
neg_violation = jnp.sum(jnp.maximum(-state.amplitude, 0.0) ** 2)

```
# Upper bound violation
upper_violation = jnp.sum(jnp.maximum(state.amplitude - bounds.max_amplitude, 0.0) ** 2)

return neg_violation + upper_violation
```

@jax.jit
def energy_violation_loss(state: WaveState, bounds: InvariantBounds = DEFAULT_BOUNDS) -> float:
“”“Loss for energy bound violations.”””
e = total_energy(state)
return jnp.maximum(e - bounds.max_energy, 0.0) ** 2

@jax.jit
def invariant_loss(state: WaveState, bounds: InvariantBounds = DEFAULT_BOUNDS) -> float:
“”“Combined loss for all invariant violations.”””
return amplitude_violation_loss(state, bounds) + energy_violation_loss(state, bounds)

# ============================================================================

# Adaptive Bounds

# ============================================================================

def scale_bounds(bounds: InvariantBounds, factor: float) -> InvariantBounds:
“”“Scale bounds by a factor (useful for curriculum learning).”””
return InvariantBounds(
min_amplitude=bounds.min_amplitude,
max_amplitude=bounds.max_amplitude * factor,
max_energy=bounds.max_energy * (factor ** 2),  # Energy scales as amplitude²
phase_gate_threshold=bounds.phase_gate_threshold
)

def bounds_for_n_modes(n_modes: int, energy_per_mode: float = 1.0) -> InvariantBounds:
“”“Create bounds appropriate for a given number of modes.”””
return InvariantBounds(
max_amplitude=jnp.sqrt(energy_per_mode * 2),  # Allow some headroom
max_energy=energy_per_mode * n_modes,
phase_gate_threshold=OMEGA
)

# ============================================================================

# Tests

# ============================================================================

def test_invariants():
“”“Test invariant bounds and stability metrics.”””
print(”=” * 60)
print(”  Invariant Bounds Tests”)
print(”=” * 60)

```
try:
    from .representation import random, WaveState
except ImportError:
    from representation import random, WaveState

key = jax.random.PRNGKey(42)

# Test 1: Default bounds
print("\n[1] Default bounds...")
bounds = DEFAULT_BOUNDS
print(f"    min_amplitude: {bounds.min_amplitude}")
print(f"    max_amplitude: {bounds.max_amplitude}")
print(f"    max_energy: {bounds.max_energy}")
print(f"    phase_gate_threshold: {bounds.phase_gate_threshold}")

# Test 2: Stability check on valid state
print("\n[2] Stability check (valid state)...")
state = WaveState(
    amplitude=jnp.array([1.0, 2.0, 1.5]),
    phase=jnp.array([0.0, 0.5, -0.5])
)
stable = is_stable(state, bounds)
metrics = measure_stability(state, bounds)
print(f"    ✓ Is stable: {stable}")
print(f"    ✓ Amplitude margin: {metrics['amplitude_margin']:.3f}")
print(f"    ✓ Energy margin: {metrics['energy_margin']:.3f}")
print(f"    ✓ Overall stability: {metrics['overall']:.3f}")
assert stable, "Valid state should be stable"

# Test 3: Stability check on invalid state (amplitude too high)
print("\n[3] Stability check (amplitude violation)...")
bad_state = WaveState(
    amplitude=jnp.array([1.0, 15.0, 1.0]),  # 15 > max_amplitude=10
    phase=jnp.array([0.0, 0.0, 0.0])
)
stable = is_stable(bad_state, bounds)
loss = amplitude_violation_loss(bad_state, bounds)
print(f"    ✓ Is stable: {stable} (expected: False)")
print(f"    ✓ Amplitude violation loss: {loss:.3f}")
assert not stable, "Invalid state should not be stable"
assert loss > 0, "Should have positive violation loss"

# Test 4: Stability check on invalid state (energy too high)
print("\n[4] Stability check (energy violation)...")
# Many modes at amplitude 5 each -> energy = 25 * n_modes
bad_state = WaveState(
    amplitude=jnp.ones(10) * 5.0,  # Energy = 250 > max_energy=100
    phase=jnp.zeros(10)
)
stable = is_stable(bad_state, bounds)
loss = energy_violation_loss(bad_state, bounds)
print(f"    ✓ Total energy: {total_energy(bad_state):.1f}")
print(f"    ✓ Is stable: {stable} (expected: False)")
print(f"    ✓ Energy violation loss: {loss:.3f}")
assert not stable, "High energy state should not be stable"
assert loss > 0, "Should have positive violation loss"

# Test 5: Scale bounds
print("\n[5] Scale bounds...")
scaled = scale_bounds(bounds, 2.0)
print(f"    Original max_amplitude: {bounds.max_amplitude}")
print(f"    Scaled max_amplitude: {scaled.max_amplitude}")
print(f"    Original max_energy: {bounds.max_energy}")
print(f"    Scaled max_energy: {scaled.max_energy}")
assert scaled.max_amplitude == bounds.max_amplitude * 2.0
assert scaled.max_energy == bounds.max_energy * 4.0  # Scales as square

# Test 6: Bounds for n_modes
print("\n[6] Bounds for n_modes...")
n_modes = 64
auto_bounds = bounds_for_n_modes(n_modes, energy_per_mode=1.0)
print(f"    n_modes: {n_modes}")
print(f"    max_amplitude: {auto_bounds.max_amplitude:.3f}")
print(f"    max_energy: {auto_bounds.max_energy:.1f}")

# Test 7: Combined invariant loss
print("\n[7] Combined invariant loss...")
# Valid state should have zero loss
valid_state = WaveState(
    amplitude=jnp.array([1.0, 1.0, 1.0]),
    phase=jnp.array([0.0, 0.0, 0.0])
)
loss = invariant_loss(valid_state, bounds)
print(f"    ✓ Valid state loss: {loss:.6f}")
assert loss < 1e-10, "Valid state should have ~zero loss"

# Invalid state should have positive loss
invalid_state = WaveState(
    amplitude=jnp.array([15.0, 15.0, 15.0]),  # Violates both bounds
    phase=jnp.array([0.0, 0.0, 0.0])
)
loss = invariant_loss(invalid_state, bounds)
print(f"    ✓ Invalid state loss: {loss:.3f}")
assert loss > 0, "Invalid state should have positive loss"

print("\n" + "=" * 60)
print("  All invariant tests passed! ✓")
print("=" * 60)

return True
```

if **name** == “**main**”:
test_invariants()
