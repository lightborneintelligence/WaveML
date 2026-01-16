"""
WaveML-JAX: WaveSeq - Wave-Native Sequence Model
=================================================
Lightborne Intelligence

WaveSeq is a recurrent architecture where:
    - State is a wave (amplitude + phase)
    - Transitions preserve wave structure
    - ERA enforces invariants at every step

Architecture:
    Input -> [WaveCell] -> ERA -> [WaveCell] -> ERA -> ... -> Output

The WaveCell performs linear wave mixing and input injection.
ERA stabilizes the trajectory after each step.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Optional, Callable
from functools import partial
import flax.linen as nn

from ..core.representation import WaveState, to_complex, from_complex, total_energy
from ..core.invariants import InvariantBounds, DEFAULT_BOUNDS
from ..core.era_rectify import era_rectify


class WaveSeqParams(NamedTuple):
    """Parameters for WaveSeq cell."""
    W_amp: jnp.ndarray
    W_phase: jnp.ndarray
    W_in_amp: jnp.ndarray
    W_in_phase: jnp.ndarray
    b_amp: jnp.ndarray
    b_phase: jnp.ndarray


def init_waveseq_params(key: jax.random.PRNGKey,
                        input_dim: int,
                        hidden_dim: int,
                        scale: float = 0.1) -> WaveSeqParams:
    """Initialize WaveSeq parameters."""
    keys = jax.random.split(key, 6)
    
    return WaveSeqParams(
        W_amp=jax.random.normal(keys[0], (hidden_dim, hidden_dim)) * scale,
        W_phase=jax.random.normal(keys[1], (hidden_dim, hidden_dim)) * scale,
        W_in_amp=jax.random.normal(keys[2], (input_dim, hidden_dim)) * scale,
        W_in_phase=jax.random.normal(keys[3], (input_dim, hidden_dim)) * scale,
        b_amp=jnp.zeros(hidden_dim),
        b_phase=jnp.zeros(hidden_dim),
    )


@jax.jit
def waveseq_step(state: WaveState,
                 x: jnp.ndarray,
                 params: WaveSeqParams,
                 bounds: InvariantBounds = DEFAULT_BOUNDS) -> WaveState:
    """
    Single WaveSeq step.
    
    Args:
        state: Current wave state (amplitude, phase)
        x: Input vector
        params: WaveSeq parameters
        bounds: ERA bounds
    
    Returns:
        Next wave state (after ERA)
    """
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
    
    new_state = WaveState(
        amplitude=jnp.abs(new_amp),
        phase=new_phase
    )
    
    return era_rectify(new_state, bounds)


def waveseq_forward(params: WaveSeqParams,
                    inputs: jnp.ndarray,
                    initial_state: Optional[WaveState] = None,
                    bounds: InvariantBounds = DEFAULT_BOUNDS) -> Tuple[WaveState, jnp.ndarray]:
    """
    Forward pass through WaveSeq.
    
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


class WaveSeqCell(nn.Module):
    """Flax module for WaveSeq cell."""
    hidden_dim: int
    bounds: InvariantBounds = DEFAULT_BOUNDS
    
    @nn.compact
    def __call__(self, state: WaveState, x: jnp.ndarray) -> WaveState:
        input_dim = x.shape[-1]
        
        new_amp = nn.tanh(
            nn.Dense(self.hidden_dim, name='amp_recurrent')(state.amplitude) +
            nn.Dense(self.hidden_dim, name='amp_input')(x)
        )
        
        new_phase = (
            nn.Dense(self.hidden_dim, name='phase_recurrent')(state.phase) +
            nn.Dense(self.hidden_dim, name='phase_input')(x)
        )
        
        new_state = WaveState(
            amplitude=jnp.abs(new_amp),
            phase=new_phase
        )
        
        return era_rectify(new_state, self.bounds)


class WaveSeq(nn.Module):
    """
    Full WaveSeq sequence model.
    
    Architecture:
        Input -> Encoder -> [WaveSeqCell + ERA]* -> Decoder -> Output
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


@jax.jit
def detect_collapse(amplitudes: jnp.ndarray, threshold: float = 0.01) -> dict:
    """
    Detect if sequence has collapsed.
    
    Args:
        amplitudes: Amplitude trajectory, shape (seq_len, hidden_dim)
        threshold: Collapse threshold
    
    Returns:
        Dict with collapse metrics
    """
    energy = jnp.sum(amplitudes ** 2, axis=-1)
    amp_var = jnp.var(amplitudes, axis=-1)
    var_collapse = jnp.mean(amp_var < threshold)
    energy_collapse = jnp.mean(energy < threshold)
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
