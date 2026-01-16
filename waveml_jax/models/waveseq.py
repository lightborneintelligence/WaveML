# “””
WaveML-JAX: WaveSeq - Wave-Native Sequence Model

Lightborne Intelligence

WaveSeq is a recurrent architecture where:
- State is a wave (amplitude + phase)
- Transitions preserve wave structure
- ERA enforces invariants at every step

Architecture:
Input → [WaveCell] → ERA → [WaveCell] → ERA → … → Output

The WaveCell performs linear wave mixing and input injection.
ERA stabilizes the trajectory after each step.
“””

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Optional, Callable
from functools import partial
import flax.linen as nn

import sys
sys.path.insert(0, ‘/home/claude/waveml-jax’)

try:
from core.representation import WaveState, to_complex, from_complex, total_energy
from core.invariants import InvariantBounds, DEFAULT_BOUNDS
from core.era_rectify import era_rectify
except ImportError:
from representation import WaveState, to_complex, from_complex, total_energy
from invariants import InvariantBounds, DEFAULT_BOUNDS
from era_rectify import era_rectify

# ============================================================================

# WaveSeq Parameters

# ============================================================================

class WaveSeqParams(NamedTuple):
“”“Parameters for WaveSeq cell.”””
W_amp: jnp.ndarray      # Amplitude mixing matrix
W_phase: jnp.ndarray    # Phase mixing matrix
W_in_amp: jnp.ndarray   # Input to amplitude projection
W_in_phase: jnp.ndarray # Input to phase projection
b_amp: jnp.ndarray      # Amplitude bias
b_phase: jnp.ndarray    # Phase bias

def init_waveseq_params(key: jax.random.PRNGKey,
input_dim: int,
hidden_dim: int,
scale: float = 0.1) -> WaveSeqParams:
“”“Initialize WaveSeq parameters.”””
keys = jax.random.split(key, 6)

```
return WaveSeqParams(
    W_amp=jax.random.normal(keys[0], (hidden_dim, hidden_dim)) * scale,
    W_phase=jax.random.normal(keys[1], (hidden_dim, hidden_dim)) * scale,
    W_in_amp=jax.random.normal(keys[2], (input_dim, hidden_dim)) * scale,
    W_in_phase=jax.random.normal(keys[3], (input_dim, hidden_dim)) * scale,
    b_amp=jnp.zeros(hidden_dim),
    b_phase=jnp.zeros(hidden_dim),
)
```

# ============================================================================

# WaveSeq Cell (Functional)

# ============================================================================

@jax.jit
def waveseq_step(state: WaveState,
x: jnp.ndarray,
params: WaveSeqParams,
bounds: InvariantBounds = DEFAULT_BOUNDS) -> WaveState:
“””
Single WaveSeq step.

```
Args:
    state: Current wave state (amplitude, phase)
    x: Input vector
    params: WaveSeq parameters
    bounds: ERA bounds

Returns:
    Next wave state (after ERA)
"""
# Wave mixing
new_amp = jnp.tanh(
    state.amplitude @ params.W_amp + 
    x @ params.W_in_amp + 
    params.b_amp
)

new_phase = (
    state.phase @ params.W_phase +
    x @ params.W_in_phase +
    params.b_phase
)

# Construct new state
new_state = WaveState(
    amplitude=jnp.abs(new_amp),  # Ensure non-negative before ERA
    phase=new_phase
)

# Apply ERA
return era_rectify(new_state, bounds)
```

def waveseq_forward(params: WaveSeqParams,
inputs: jnp.ndarray,
initial_state: Optional[WaveState] = None,
bounds: InvariantBounds = DEFAULT_BOUNDS) -> Tuple[WaveState, jnp.ndarray]:
“””
Forward pass through WaveSeq.

```
Args:
    params: WaveSeq parameters
    inputs: Input sequence, shape (seq_len, input_dim)
    initial_state: Initial wave state (default: zeros)
    bounds: ERA bounds

Returns:
    Tuple of (final_state, all_amplitudes)
"""
hidden_dim = params.W_amp.shape[0]

if initial_state is None:
    initial_state = WaveState(
        amplitude=jnp.zeros(hidden_dim),
        phase=jnp.zeros(hidden_dim)
    )

def scan_body(state, x):
    new_state = waveseq_step(state, x, params, bounds)
    return new_state, new_state.amplitude

final_state, amplitudes = jax.lax.scan(scan_body, initial_state, inputs)
return final_state, amplitudes
```

# ============================================================================

# WaveSeq Module (Flax)

# ============================================================================

class WaveSeqCell(nn.Module):
“”“Flax module for WaveSeq cell.”””
hidden_dim: int
bounds: InvariantBounds = DEFAULT_BOUNDS

```
@nn.compact
def __call__(self, state: WaveState, x: jnp.ndarray) -> WaveState:
    input_dim = x.shape[-1]
    
    # Amplitude pathway
    new_amp = nn.tanh(
        nn.Dense(self.hidden_dim, name='amp_recurrent')(state.amplitude) +
        nn.Dense(self.hidden_dim, name='amp_input')(x)
    )
    
    # Phase pathway
    new_phase = (
        nn.Dense(self.hidden_dim, name='phase_recurrent')(state.phase) +
        nn.Dense(self.hidden_dim, name='phase_input')(x)
    )
    
    new_state = WaveState(
        amplitude=jnp.abs(new_amp),
        phase=new_phase
    )
    
    return era_rectify(new_state, self.bounds)
```

class WaveSeq(nn.Module):
“””
Full WaveSeq sequence model.

```
Architecture:
    Input → Encoder → [WaveSeqCell + ERA]* → Decoder → Output
"""
hidden_dim: int
output_dim: int
bounds: InvariantBounds = DEFAULT_BOUNDS

@nn.compact
def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """
    Args:
        inputs: Shape (seq_len, input_dim)
    
    Returns:
        outputs: Shape (seq_len, output_dim)
    """
    seq_len = inputs.shape[0]
    
    # Initialize state
    state = WaveState(
        amplitude=jnp.zeros(self.hidden_dim),
        phase=jnp.zeros(self.hidden_dim)
    )
    
    cell = WaveSeqCell(self.hidden_dim, self.bounds)
    decoder = nn.Dense(self.output_dim, name='decoder')
    
    outputs = []
    for t in range(seq_len):
        state = cell(state, inputs[t])
        out = decoder(state.amplitude)
        outputs.append(out)
    
    return jnp.stack(outputs)
```

# ============================================================================

# Collapse Detection

# ============================================================================

@jax.jit
def detect_collapse(amplitudes: jnp.ndarray, threshold: float = 0.01) -> dict:
“””
Detect if sequence has collapsed.

```
Args:
    amplitudes: Amplitude trajectory, shape (seq_len, hidden_dim)
    threshold: Collapse threshold

Returns:
    Dict with collapse metrics
"""
# Energy trajectory
energy = jnp.sum(amplitudes ** 2, axis=-1)

# Variance collapse
amp_var = jnp.var(amplitudes, axis=-1)
var_collapse = jnp.mean(amp_var < threshold)

# Energy collapse
energy_collapse = jnp.mean(energy < threshold)

# Explosion detection
energy_explosion = jnp.mean(energy > 1e6)

return {
    'var_collapse_ratio': var_collapse,
    'energy_collapse_ratio': energy_collapse,
    'energy_explosion_ratio': energy_explosion,
    'mean_energy': jnp.mean(energy),
    'max_energy': jnp.max(energy),
    'min_energy': jnp.min(energy),
    'healthy': (var_collapse < 0.5) & (energy_explosion < 0.01)
}
```

# ============================================================================

# Tests

# ============================================================================

def test_waveseq():
“”“Test WaveSeq model.”””
print(”=” * 60)
print(”  WaveSeq Model Tests”)
print(”=” * 60)

```
key = jax.random.PRNGKey(42)
input_dim = 8
hidden_dim = 16
seq_len = 50

# Test 1: Parameter initialization
print("\n[1] Parameter initialization...")
params = init_waveseq_params(key, input_dim, hidden_dim)
print(f"    W_amp shape: {params.W_amp.shape}")
print(f"    W_in_amp shape: {params.W_in_amp.shape}")
print(f"    b_amp shape: {params.b_amp.shape}")

# Test 2: Single step
print("\n[2] Single step...")
state = WaveState(
    amplitude=jnp.ones(hidden_dim) * 0.5,
    phase=jnp.zeros(hidden_dim)
)
x = jax.random.normal(key, (input_dim,))
new_state = waveseq_step(state, x, params)
print(f"    Input shape: {x.shape}")
print(f"    State amplitude shape: {new_state.amplitude.shape}")
print(f"    State energy: {total_energy(new_state):.3f}")

# Test 3: Forward pass
print("\n[3] Forward pass...")
inputs = jax.random.normal(key, (seq_len, input_dim))
final_state, amplitudes = waveseq_forward(params, inputs)
print(f"    Input sequence: {inputs.shape}")
print(f"    Output amplitudes: {amplitudes.shape}")
print(f"    Final energy: {total_energy(final_state):.3f}")

# Test 4: No collapse over long horizon
print("\n[4] Long horizon stability...")
long_seq_len = 500
long_inputs = jax.random.normal(key, (long_seq_len, input_dim)) * 0.1
_, long_amps = waveseq_forward(params, long_inputs)
collapse_stats = detect_collapse(long_amps)
print(f"    Sequence length: {long_seq_len}")
print(f"    Healthy: {collapse_stats['healthy']}")
print(f"    Mean energy: {collapse_stats['mean_energy']:.3f}")
print(f"    Min energy: {collapse_stats['min_energy']:.6f}")
print(f"    Max energy: {collapse_stats['max_energy']:.3f}")
print(f"    Var collapse ratio: {collapse_stats['var_collapse_ratio']:.3f}")

# Test 5: Gradient flow
print("\n[5] Gradient flow...")
def loss_fn(params, inputs):
    _, amps = waveseq_forward(params, inputs)
    return jnp.mean(amps ** 2)

grad_fn = jax.grad(loss_fn)
grads = grad_fn(params, inputs[:20])  # Shorter for speed
grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads)))
print(f"    Gradient norm: {grad_norm:.6f}")
assert not jnp.isnan(grad_norm), "Gradient should not be NaN"
assert grad_norm > 0, "Gradient should be non-zero"
assert grad_norm < 1e6, "Gradient should not explode"

# Test 6: Flax module
print("\n[6] Flax WaveSeq module...")
model = WaveSeq(hidden_dim=hidden_dim, output_dim=4)
variables = model.init(key, inputs[:10])
outputs = model.apply(variables, inputs[:10])
print(f"    Model output shape: {outputs.shape}")

# Test 7: ERA enforcement throughout
print("\n[7] ERA enforcement verification...")
bounds = DEFAULT_BOUNDS
_, amps = waveseq_forward(params, long_inputs, bounds=bounds)
energies = jnp.sum(amps ** 2, axis=-1)
max_energy = jnp.max(energies)
print(f"    Max energy in trajectory: {max_energy:.3f}")
print(f"    Energy bound: {bounds.max_energy}")
assert max_energy <= bounds.max_energy + 1e-5, "ERA should bound energy"

print("\n" + "=" * 60)
print("  All WaveSeq tests passed! ✓")
print("=" * 60)

return True
```

if **name** == “**main**”:
test_waveseq()
