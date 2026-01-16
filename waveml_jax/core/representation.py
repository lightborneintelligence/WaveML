# “””
WaveML-JAX: Wave State Representation

Lightborne Intelligence

The fundamental data structure for wave-native computation.
Wave state separates amplitude (energy) from phase (alignment).

Complex form: Ψ = A · e^(iφ)
“””

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from functools import partial

class WaveState(NamedTuple):
“””
Wave state representation.

```
Attributes:
    amplitude: Non-negative energy magnitude, shape (*, n_modes)
    phase: Wrapped to [-π, π], shape (*, n_modes)
"""
amplitude: jnp.ndarray
phase: jnp.ndarray

@property
def shape(self) -> Tuple[int, ...]:
    return self.amplitude.shape

@property
def n_modes(self) -> int:
    return self.amplitude.shape[-1]
```

# ============================================================================

# Complex Conversion

# ============================================================================

@jax.jit
def to_complex(state: WaveState) -> jnp.ndarray:
“”“Convert wave state to complex representation: A * e^(iφ)”””
return state.amplitude * jnp.exp(1j * state.phase)

@jax.jit
def from_complex(z: jnp.ndarray) -> WaveState:
“”“Convert complex array to wave state.”””
amplitude = jnp.abs(z)
phase = jnp.angle(z)
return WaveState(amplitude=amplitude, phase=phase)

# ============================================================================

# Real Interleaved Conversion

# ============================================================================

@jax.jit
def to_real(state: WaveState) -> jnp.ndarray:
“”“Convert to real interleaved format: [a0, φ0, a1, φ1, …]”””
# Stack and interleave
stacked = jnp.stack([state.amplitude, state.phase], axis=-1)
return stacked.reshape(*state.amplitude.shape[:-1], -1)

@jax.jit
def from_real(x: jnp.ndarray) -> WaveState:
“”“Convert from real interleaved format.”””
# Reshape to (…, n_modes, 2)
reshaped = x.reshape(*x.shape[:-1], -1, 2)
amplitude = reshaped[…, 0]
phase = reshaped[…, 1]
return WaveState(amplitude=amplitude, phase=phase)

# ============================================================================

# FFT Encoding/Decoding

# ============================================================================

def encode_fft(signal: jnp.ndarray, n_modes: int = None) -> WaveState:
“””
Encode real signal to wave state via FFT.

```
Args:
    signal: Real-valued signal, shape (..., length)
    n_modes: Number of frequency modes (default: length // 2 + 1)

Returns:
    WaveState with amplitude and phase
"""
spectrum = jnp.fft.rfft(signal, axis=-1)

if n_modes is not None:
    spectrum = spectrum[..., :n_modes]

return from_complex(spectrum)
```

def decode_fft(state: WaveState, length: int) -> jnp.ndarray:
“””
Decode wave state to real signal via inverse FFT.

```
Args:
    state: WaveState to decode
    length: Output signal length

Returns:
    Real-valued signal
"""
spectrum = to_complex(state)
return jnp.fft.irfft(spectrum, n=length, axis=-1)
```

# ============================================================================

# Energy Metrics

# ============================================================================

@jax.jit
def energy(state: WaveState) -> jnp.ndarray:
“”“Per-mode energy: A²”””
return state.amplitude ** 2

@jax.jit
def total_energy(state: WaveState) -> jnp.ndarray:
“”“Total energy: Σ A²”””
return jnp.sum(energy(state), axis=-1)

@jax.jit
def phase_coherence(state: WaveState) -> jnp.ndarray:
“””
Phase coherence measure.

```
Returns value in [0, 1] where 1 = perfectly aligned phases.
Weighted by amplitude (low-energy modes contribute less).
"""
# Complex unit vectors
unit_phasors = jnp.exp(1j * state.phase)

# Amplitude-weighted mean
weights = state.amplitude / (jnp.sum(state.amplitude, axis=-1, keepdims=True) + 1e-8)
mean_phasor = jnp.sum(weights * unit_phasors, axis=-1)

return jnp.abs(mean_phasor)
```

# ============================================================================

# Initialization

# ============================================================================

def zeros(n_modes: int, batch_shape: Tuple[int, …] = ()) -> WaveState:
“”“Create zero-initialized wave state.”””
shape = batch_shape + (n_modes,)
return WaveState(
amplitude=jnp.zeros(shape),
phase=jnp.zeros(shape)
)

def ones(n_modes: int, batch_shape: Tuple[int, …] = ()) -> WaveState:
“”“Create unit-amplitude, zero-phase wave state.”””
shape = batch_shape + (n_modes,)
return WaveState(
amplitude=jnp.ones(shape),
phase=jnp.zeros(shape)
)

def random(key: jax.random.PRNGKey, n_modes: int,
batch_shape: Tuple[int, …] = (),
amp_scale: float = 1.0) -> WaveState:
“”“Create random wave state.”””
shape = batch_shape + (n_modes,)
k1, k2 = jax.random.split(key)

```
amplitude = jax.random.uniform(k1, shape) * amp_scale
phase = jax.random.uniform(k2, shape, minval=-jnp.pi, maxval=jnp.pi)

return WaveState(amplitude=amplitude, phase=phase)
```

# ============================================================================

# Tests

# ============================================================================

def test_representation():
“”“Test wave state representation.”””
print(”=” * 60)
print(”  WaveState Representation Tests”)
print(”=” * 60)

```
key = jax.random.PRNGKey(42)
n_modes = 16

# Test 1: Creation and properties
print("\n[1] Creation and properties...")
state = random(key, n_modes, amp_scale=2.0)
assert state.shape == (n_modes,), f"Shape mismatch: {state.shape}"
assert state.n_modes == n_modes, f"n_modes mismatch: {state.n_modes}"
print(f"    ✓ Shape: {state.shape}, n_modes: {state.n_modes}")

# Test 2: Complex roundtrip
print("\n[2] Complex roundtrip...")
z = to_complex(state)
state_back = from_complex(z)
amp_err = jnp.max(jnp.abs(state.amplitude - state_back.amplitude))
phase_err = jnp.max(jnp.abs(state.phase - state_back.phase))
assert amp_err < 1e-6, f"Amplitude error: {amp_err}"
assert phase_err < 1e-6, f"Phase error: {phase_err}"
print(f"    ✓ Amplitude error: {amp_err:.2e}")
print(f"    ✓ Phase error: {phase_err:.2e}")

# Test 3: Real interleaved roundtrip
print("\n[3] Real interleaved roundtrip...")
x = to_real(state)
state_back = from_real(x)
amp_err = jnp.max(jnp.abs(state.amplitude - state_back.amplitude))
phase_err = jnp.max(jnp.abs(state.phase - state_back.phase))
assert amp_err < 1e-6, f"Amplitude error: {amp_err}"
assert phase_err < 1e-6, f"Phase error: {phase_err}"
print(f"    ✓ Real vector length: {x.shape[-1]} (2 × {n_modes})")
print(f"    ✓ Roundtrip error: {max(amp_err, phase_err):.2e}")

# Test 4: FFT encoding/decoding
print("\n[4] FFT encoding/decoding...")
signal = jnp.sin(jnp.linspace(0, 4 * jnp.pi, 64))
encoded = encode_fft(signal)
decoded = decode_fft(encoded, len(signal))
fft_err = jnp.max(jnp.abs(signal - decoded))
assert fft_err < 1e-5, f"FFT roundtrip error: {fft_err}"
print(f"    ✓ Signal length: {len(signal)}")
print(f"    ✓ Encoded modes: {encoded.n_modes}")
print(f"    ✓ Roundtrip error: {fft_err:.2e}")

# Test 5: Energy metrics
print("\n[5] Energy metrics...")
state = WaveState(
    amplitude=jnp.array([1.0, 2.0, 3.0]),
    phase=jnp.array([0.0, 0.0, 0.0])
)
e = total_energy(state)
expected = 1.0 + 4.0 + 9.0  # 14.0
assert jnp.abs(e - expected) < 1e-6, f"Energy error: {e} != {expected}"
print(f"    ✓ Total energy: {e:.2f} (expected: {expected})")

# Test 6: Phase coherence
print("\n[6] Phase coherence...")
# All phases aligned -> coherence = 1
aligned = WaveState(
    amplitude=jnp.ones(8),
    phase=jnp.zeros(8)
)
coh = phase_coherence(aligned)
assert coh > 0.99, f"Aligned coherence should be ~1: {coh}"
print(f"    ✓ Aligned phases coherence: {coh:.4f}")

# Random phases -> lower coherence
key = jax.random.PRNGKey(123)
scattered = random(key, 64)
coh = phase_coherence(scattered)
print(f"    ✓ Random phases coherence: {coh:.4f}")

# Test 7: Batched operations
print("\n[7] Batched operations...")
batch_state = random(key, n_modes, batch_shape=(4, 8))
assert batch_state.shape == (4, 8, n_modes)
z_batch = to_complex(batch_state)
assert z_batch.shape == (4, 8, n_modes)
e_batch = total_energy(batch_state)
assert e_batch.shape == (4, 8)
print(f"    ✓ Batch shape: {batch_state.shape}")
print(f"    ✓ Complex shape: {z_batch.shape}")
print(f"    ✓ Energy shape: {e_batch.shape}")

print("\n" + "=" * 60)
print("  All representation tests passed! ✓")
print("=" * 60)

return True
```

if **name** == “**main**”:
test_representation()
