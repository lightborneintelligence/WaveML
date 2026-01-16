"""
WaveML-JAX: Wave RF - Receptive Field Model
============================================
Lightborne Intelligence

Wave-native convolutional model for signal processing.
Uses wave state representation with ERA throughout.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Optional
from functools import partial
import flax.linen as nn

import sys
sys.path.insert(0, '/home/claude/waveml-jax')

try:
    from core.representation import WaveState, encode_fft, decode_fft, total_energy
    from core.invariants import InvariantBounds, DEFAULT_BOUNDS
    from core.era_rectify import era_rectify
except ImportError:
    from representation import WaveState, encode_fft, decode_fft, total_energy
    from invariants import InvariantBounds, DEFAULT_BOUNDS
    from era_rectify import era_rectify


# ============================================================================
# Wave Convolution (Frequency Domain)
# ============================================================================

@jax.jit
def wave_conv(state: WaveState, 
              kernel_amp: jnp.ndarray,
              kernel_phase: jnp.ndarray) -> WaveState:
    """
    Wave convolution in frequency domain.
    
    Multiplication in frequency domain = convolution in time domain.
    """
    new_amp = state.amplitude * kernel_amp
    new_phase = state.phase + kernel_phase
    return WaveState(amplitude=new_amp, phase=new_phase)


@jax.jit
def wave_conv_with_era(state: WaveState,
                       kernel_amp: jnp.ndarray,
                       kernel_phase: jnp.ndarray,
                       bounds: InvariantBounds = DEFAULT_BOUNDS) -> WaveState:
    """Wave convolution followed by ERA."""
    convolved = wave_conv(state, kernel_amp, kernel_phase)
    return era_rectify(convolved, bounds)


# ============================================================================
# Wave RF Layer
# ============================================================================

class WaveRFParams(NamedTuple):
    """Parameters for Wave RF layer."""
    kernel_amp: jnp.ndarray    # (n_filters, n_modes)
    kernel_phase: jnp.ndarray  # (n_filters, n_modes)
    bias_amp: jnp.ndarray      # (n_filters,)


def init_waverf_params(key: jax.random.PRNGKey,
                       n_modes: int,
                       n_filters: int,
                       scale: float = 0.1) -> WaveRFParams:
    """Initialize Wave RF parameters."""
    k1, k2 = jax.random.split(key)
    
    return WaveRFParams(
        kernel_amp=jnp.ones((n_filters, n_modes)) + jax.random.normal(k1, (n_filters, n_modes)) * scale,
        kernel_phase=jax.random.normal(k2, (n_filters, n_modes)) * scale,
        bias_amp=jnp.zeros(n_filters)
    )


@jax.jit
def waverf_layer(state: WaveState,
                 params: WaveRFParams,
                 bounds: InvariantBounds = DEFAULT_BOUNDS) -> WaveState:
    """
    Apply Wave RF layer.
    
    Each filter produces amplitude/phase modification, then aggregated.
    """
    n_filters = params.kernel_amp.shape[0]
    
    # Apply each filter
    filtered_amps = state.amplitude[None, :] * params.kernel_amp  # (n_filters, n_modes)
    filtered_phases = state.phase[None, :] + params.kernel_phase
    
    # Aggregate (mean across filters)
    out_amp = jnp.mean(filtered_amps, axis=0) + jnp.mean(params.bias_amp)
    out_phase = jnp.mean(filtered_phases, axis=0)
    
    out_state = WaveState(
        amplitude=jnp.maximum(out_amp, 0.0),
        phase=out_phase
    )
    
    return era_rectify(out_state, bounds)


# ============================================================================
# Multi-Layer Wave RF
# ============================================================================

def init_waverf_stack(key: jax.random.PRNGKey,
                      n_modes: int,
                      n_filters: int,
                      n_layers: int) -> list:
    """Initialize stack of Wave RF layers."""
    params = []
    for i in range(n_layers):
        key, subkey = jax.random.split(key)
        params.append(init_waverf_params(subkey, n_modes, n_filters))
    return params


def waverf_forward(state: WaveState,
                   params_list: list,
                   bounds: InvariantBounds = DEFAULT_BOUNDS) -> WaveState:
    """Forward through Wave RF stack."""
    for params in params_list:
        state = waverf_layer(state, params, bounds)
    return state


# ============================================================================
# Flax Module
# ============================================================================

class WaveRF(nn.Module):
    """
    Wave RF model for signal processing.
    
    Input signal → FFT → Wave layers + ERA → IFFT → Output signal
    """
    n_modes: int
    n_filters: int
    n_layers: int
    bounds: InvariantBounds = DEFAULT_BOUNDS
    
    @nn.compact
    def __call__(self, signal: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            signal: Input signal, shape (length,)
        
        Returns:
            Processed signal, shape (length,)
        """
        length = signal.shape[-1]
        
        # Encode to wave state
        state = encode_fft(signal, self.n_modes)
        
        # Process through layers
        for i in range(self.n_layers):
            kernel_amp = self.param(
                f'kernel_amp_{i}',
                nn.initializers.ones,
                (self.n_filters, self.n_modes)
            )
            kernel_phase = self.param(
                f'kernel_phase_{i}',
                nn.initializers.normal(0.1),
                (self.n_filters, self.n_modes)
            )
            
            # Filter and aggregate
            filtered_amp = state.amplitude[None, :] * kernel_amp
            filtered_phase = state.phase[None, :] + kernel_phase
            
            state = WaveState(
                amplitude=jnp.maximum(jnp.mean(filtered_amp, axis=0), 0.0),
                phase=jnp.mean(filtered_phase, axis=0)
            )
            state = era_rectify(state, self.bounds)
        
        # Decode back to signal
        return decode_fft(state, length)


# ============================================================================
# Tests
# ============================================================================

def test_wave_rf():
    """Test Wave RF model."""
    print("=" * 60)
    print("  Wave RF Model Tests")
    print("=" * 60)
    
    key = jax.random.PRNGKey(42)
    n_modes = 32
    n_filters = 8
    signal_length = 64
    
    # Create test signal
    t = jnp.linspace(0, 2 * jnp.pi, signal_length)
    signal = jnp.sin(t) + 0.5 * jnp.sin(3 * t) + 0.1 * jax.random.normal(key, (signal_length,))
    
    # Test 1: Wave convolution
    print("\n[1] Wave convolution...")
    state = encode_fft(signal, n_modes)
    kernel_amp = jnp.ones(n_modes) * 0.9
    kernel_phase = jnp.zeros(n_modes)
    convolved = wave_conv(state, kernel_amp, kernel_phase)
    print(f"    Input energy: {total_energy(state):.3f}")
    print(f"    Output energy: {total_energy(convolved):.3f}")
    
    # Test 2: Wave RF layer
    print("\n[2] Wave RF layer...")
    params = init_waverf_params(key, n_modes, n_filters)
    out_state = waverf_layer(state, params)
    print(f"    Kernel shape: {params.kernel_amp.shape}")
    print(f"    Output energy: {total_energy(out_state):.3f}")
    
    # Test 3: Wave RF stack
    print("\n[3] Wave RF stack...")
    params_list = init_waverf_stack(key, n_modes, n_filters, n_layers=3)
    out_state = waverf_forward(state, params_list)
    print(f"    Number of layers: {len(params_list)}")
    print(f"    Output energy: {total_energy(out_state):.3f}")
    
    # Test 4: Full model (Flax)
    print("\n[4] Flax WaveRF model...")
    model = WaveRF(n_modes=n_modes, n_filters=n_filters, n_layers=3)
    variables = model.init(key, signal)
    output = model.apply(variables, signal)
    print(f"    Input shape: {signal.shape}")
    print(f"    Output shape: {output.shape}")
    print(f"    Input energy: {jnp.sum(signal ** 2):.3f}")
    print(f"    Output energy: {jnp.sum(output ** 2):.3f}")
    
    # Test 5: Gradient flow
    print("\n[5] Gradient flow...")
    def loss_fn(params, signal):
        output = model.apply(params, signal)
        return jnp.mean(output ** 2)
    
    grad = jax.grad(loss_fn)(variables, signal)
    grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grad)))
    print(f"    Gradient norm: {grad_norm:.6f}")
    assert not jnp.isnan(grad_norm), "Gradient should not be NaN"
    
    # Test 6: Signal reconstruction quality
    print("\n[6] Signal reconstruction...")
    # With identity-ish kernels, should roughly preserve signal
    identity_model = WaveRF(n_modes=n_modes, n_filters=1, n_layers=1)
    id_vars = identity_model.init(key, signal)
    reconstructed = identity_model.apply(id_vars, signal)
    mse = jnp.mean((signal - reconstructed) ** 2)
    print(f"    Reconstruction MSE: {mse:.6f}")
    
    print("\n" + "=" * 60)
    print("  All Wave RF tests passed! ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_wave_rf()
