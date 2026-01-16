"""
WaveML-JAX: Baseline Models
===========================
Lightborne Intelligence

Standard recurrent architectures for comparison with WaveSeq.
These demonstrate collapse behavior that ERA prevents.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Optional
from functools import partial
import flax.linen as nn


class RNNParams(NamedTuple):
    """Parameters for vanilla RNN."""
    W_h: jnp.ndarray
    W_x: jnp.ndarray
    b: jnp.ndarray


def init_rnn_params(key: jax.random.PRNGKey,
                    input_dim: int,
                    hidden_dim: int,
                    scale: float = 0.1) -> RNNParams:
    """Initialize RNN parameters."""
    k1, k2 = jax.random.split(key)
    return RNNParams(
        W_h=jax.random.normal(k1, (hidden_dim, hidden_dim)) * scale,
        W_x=jax.random.normal(k2, (input_dim, hidden_dim)) * scale,
        b=jnp.zeros(hidden_dim)
    )


@jax.jit
def rnn_step(h: jnp.ndarray, x: jnp.ndarray, params: RNNParams) -> jnp.ndarray:
    """Single RNN step: h' = tanh(W_h @ h + W_x @ x + b)"""
    return jnp.tanh(h @ params.W_h + x @ params.W_x + params.b)


def rnn_forward(params: RNNParams,
                inputs: jnp.ndarray,
                h0: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward pass through RNN."""
    hidden_dim = params.W_h.shape[0]
    if h0 is None:
        h0 = jnp.zeros(hidden_dim)
    
    def scan_body(h, x):
        h_new = rnn_step(h, x, params)
        return h_new, h_new
    
    final_h, hiddens = jax.lax.scan(scan_body, h0, inputs)
    return final_h, hiddens


class LSTMParams(NamedTuple):
    """Parameters for LSTM."""
    W_i: jnp.ndarray
    W_f: jnp.ndarray
    W_o: jnp.ndarray
    W_c: jnp.ndarray
    U_i: jnp.ndarray
    U_f: jnp.ndarray
    U_o: jnp.ndarray
    U_c: jnp.ndarray
    b_i: jnp.ndarray
    b_f: jnp.ndarray
    b_o: jnp.ndarray
    b_c: jnp.ndarray


class LSTMState(NamedTuple):
    """LSTM hidden state."""
    h: jnp.ndarray
    c: jnp.ndarray


def init_lstm_params(key: jax.random.PRNGKey,
                     input_dim: int,
                     hidden_dim: int,
                     scale: float = 0.1) -> LSTMParams:
    """Initialize LSTM parameters."""
    keys = jax.random.split(key, 8)
    
    return LSTMParams(
        W_i=jax.random.normal(keys[0], (input_dim, hidden_dim)) * scale,
        W_f=jax.random.normal(keys[1], (input_dim, hidden_dim)) * scale,
        W_o=jax.random.normal(keys[2], (input_dim, hidden_dim)) * scale,
        W_c=jax.random.normal(keys[3], (input_dim, hidden_dim)) * scale,
        U_i=jax.random.normal(keys[4], (hidden_dim, hidden_dim)) * scale,
        U_f=jax.random.normal(keys[5], (hidden_dim, hidden_dim)) * scale,
        U_o=jax.random.normal(keys[6], (hidden_dim, hidden_dim)) * scale,
        U_c=jax.random.normal(keys[7], (hidden_dim, hidden_dim)) * scale,
        b_i=jnp.zeros(hidden_dim),
        b_f=jnp.ones(hidden_dim),
        b_o=jnp.zeros(hidden_dim),
        b_c=jnp.zeros(hidden_dim),
    )


@jax.jit
def lstm_step(state: LSTMState, x: jnp.ndarray, params: LSTMParams) -> LSTMState:
    """Single LSTM step."""
    h, c = state.h, state.c
    
    i = jax.nn.sigmoid(x @ params.W_i + h @ params.U_i + params.b_i)
    f = jax.nn.sigmoid(x @ params.W_f + h @ params.U_f + params.b_f)
    o = jax.nn.sigmoid(x @ params.W_o + h @ params.U_o + params.b_o)
    c_tilde = jnp.tanh(x @ params.W_c + h @ params.U_c + params.b_c)
    
    c_new = f * c + i * c_tilde
    h_new = o * jnp.tanh(c_new)
    
    return LSTMState(h=h_new, c=c_new)


def lstm_forward(params: LSTMParams,
                 inputs: jnp.ndarray,
                 state0: Optional[LSTMState] = None) -> Tuple[LSTMState, jnp.ndarray]:
    """Forward pass through LSTM."""
    hidden_dim = params.U_i.shape[0]
    if state0 is None:
        state0 = LSTMState(
            h=jnp.zeros(hidden_dim),
            c=jnp.zeros(hidden_dim)
        )
    
    def scan_body(state, x):
        new_state = lstm_step(state, x, params)
        return new_state, new_state.h
    
    final_state, hiddens = jax.lax.scan(scan_body, state0, inputs)
    return final_state, hiddens


class GRUParams(NamedTuple):
    """Parameters for GRU."""
    W_z: jnp.ndarray
    W_r: jnp.ndarray
    W_h: jnp.ndarray
    U_z: jnp.ndarray
    U_r: jnp.ndarray
    U_h: jnp.ndarray
    b_z: jnp.ndarray
    b_r: jnp.ndarray
    b_h: jnp.ndarray


def init_gru_params(key: jax.random.PRNGKey,
                    input_dim: int,
                    hidden_dim: int,
                    scale: float = 0.1) -> GRUParams:
    """Initialize GRU parameters."""
    keys = jax.random.split(key, 6)
    
    return GRUParams(
        W_z=jax.random.normal(keys[0], (input_dim, hidden_dim)) * scale,
        W_r=jax.random.normal(keys[1], (input_dim, hidden_dim)) * scale,
        W_h=jax.random.normal(keys[2], (input_dim, hidden_dim)) * scale,
        U_z=jax.random.normal(keys[3], (hidden_dim, hidden_dim)) * scale,
        U_r=jax.random.normal(keys[4], (hidden_dim, hidden_dim)) * scale,
        U_h=jax.random.normal(keys[5], (hidden_dim, hidden_dim)) * scale,
        b_z=jnp.zeros(hidden_dim),
        b_r=jnp.zeros(hidden_dim),
        b_h=jnp.zeros(hidden_dim),
    )


@jax.jit
def gru_step(h: jnp.ndarray, x: jnp.ndarray, params: GRUParams) -> jnp.ndarray:
    """Single GRU step."""
    z = jax.nn.sigmoid(x @ params.W_z + h @ params.U_z + params.b_z)
    r = jax.nn.sigmoid(x @ params.W_r + h @ params.U_r + params.b_r)
    h_tilde = jnp.tanh(x @ params.W_h + (r * h) @ params.U_h + params.b_h)
    h_new = (1 - z) * h + z * h_tilde
    return h_new


def gru_forward(params: GRUParams,
                inputs: jnp.ndarray,
                h0: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward pass through GRU."""
    hidden_dim = params.U_z.shape[0]
    if h0 is None:
        h0 = jnp.zeros(hidden_dim)
    
    def scan_body(h, x):
        h_new = gru_step(h, x, params)
        return h_new, h_new
    
    final_h, hiddens = jax.lax.scan(scan_body, h0, inputs)
    return final_h, hiddens


def detect_baseline_collapse(hiddens: jnp.ndarray) -> dict:
    """Detect collapse/explosion in baseline models."""
    var_per_step = jnp.var(hiddens, axis=-1)
    norm_per_step = jnp.linalg.norm(hiddens, axis=-1)
    
    return {
        'mean_variance': float(jnp.mean(var_per_step)),
        'min_variance': float(jnp.min(var_per_step)),
        'max_norm': float(jnp.max(norm_per_step)),
        'final_norm': float(norm_per_step[-1]),
        'collapsed': bool(jnp.min(var_per_step) < 1e-6),
        'exploded': bool(jnp.max(norm_per_step) > 1e6),
    }
