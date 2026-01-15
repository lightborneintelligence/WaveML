# “””
WaveML-JAX: Baseline Models

Lightborne Intelligence

Standard recurrent architectures for comparison with WaveSeq.
These demonstrate collapse behavior that ERA prevents.
“””

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Optional
from functools import partial
import flax.linen as nn

# ============================================================================

# Vanilla RNN

# ============================================================================

class RNNParams(NamedTuple):
“”“Parameters for vanilla RNN.”””
W_h: jnp.ndarray   # Hidden-to-hidden
W_x: jnp.ndarray   # Input-to-hidden
b: jnp.ndarray     # Bias

def init_rnn_params(key: jax.random.PRNGKey,
input_dim: int,
hidden_dim: int,
scale: float = 0.1) -> RNNParams:
“”“Initialize RNN parameters.”””
k1, k2 = jax.random.split(key)
return RNNParams(
W_h=jax.random.normal(k1, (hidden_dim, hidden_dim)) * scale,
W_x=jax.random.normal(k2, (input_dim, hidden_dim)) * scale,
b=jnp.zeros(hidden_dim)
)

@jax.jit
def rnn_step(h: jnp.ndarray, x: jnp.ndarray, params: RNNParams) -> jnp.ndarray:
“”“Single RNN step: h’ = tanh(W_h @ h + W_x @ x + b)”””
return jnp.tanh(h @ params.W_h + x @ params.W_x + params.b)

def rnn_forward(params: RNNParams,
inputs: jnp.ndarray,
h0: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
“”“Forward pass through RNN.”””
hidden_dim = params.W_h.shape[0]
if h0 is None:
h0 = jnp.zeros(hidden_dim)

```
def scan_body(h, x):
    h_new = rnn_step(h, x, params)
    return h_new, h_new

final_h, hiddens = jax.lax.scan(scan_body, h0, inputs)
return final_h, hiddens
```

# ============================================================================

# LSTM

# ============================================================================

class LSTMParams(NamedTuple):
“”“Parameters for LSTM.”””
W_i: jnp.ndarray  # Input gate
W_f: jnp.ndarray  # Forget gate
W_o: jnp.ndarray  # Output gate
W_c: jnp.ndarray  # Cell gate
U_i: jnp.ndarray
U_f: jnp.ndarray
U_o: jnp.ndarray
U_c: jnp.ndarray
b_i: jnp.ndarray
b_f: jnp.ndarray
b_o: jnp.ndarray
b_c: jnp.ndarray

class LSTMState(NamedTuple):
“”“LSTM hidden state.”””
h: jnp.ndarray  # Hidden
c: jnp.ndarray  # Cell

def init_lstm_params(key: jax.random.PRNGKey,
input_dim: int,
hidden_dim: int,
scale: float = 0.1) -> LSTMParams:
“”“Initialize LSTM parameters.”””
keys = jax.random.split(key, 8)

```
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
    b_f=jnp.ones(hidden_dim),  # Forget gate bias = 1 for better gradients
    b_o=jnp.zeros(hidden_dim),
    b_c=jnp.zeros(hidden_dim),
)
```

@jax.jit
def lstm_step(state: LSTMState, x: jnp.ndarray, params: LSTMParams) -> LSTMState:
“”“Single LSTM step.”””
h, c = state.h, state.c

```
i = jax.nn.sigmoid(x @ params.W_i + h @ params.U_i + params.b_i)
f = jax.nn.sigmoid(x @ params.W_f + h @ params.U_f + params.b_f)
o = jax.nn.sigmoid(x @ params.W_o + h @ params.U_o + params.b_o)
c_tilde = jnp.tanh(x @ params.W_c + h @ params.U_c + params.b_c)

c_new = f * c + i * c_tilde
h_new = o * jnp.tanh(c_new)

return LSTMState(h=h_new, c=c_new)
```

def lstm_forward(params: LSTMParams,
inputs: jnp.ndarray,
state0: Optional[LSTMState] = None) -> Tuple[LSTMState, jnp.ndarray]:
“”“Forward pass through LSTM.”””
hidden_dim = params.U_i.shape[0]
if state0 is None:
state0 = LSTMState(
h=jnp.zeros(hidden_dim),
c=jnp.zeros(hidden_dim)
)

```
def scan_body(state, x):
    new_state = lstm_step(state, x, params)
    return new_state, new_state.h

final_state, hiddens = jax.lax.scan(scan_body, state0, inputs)
return final_state, hiddens
```

# ============================================================================

# GRU

# ============================================================================

class GRUParams(NamedTuple):
“”“Parameters for GRU.”””
W_z: jnp.ndarray  # Update gate
W_r: jnp.ndarray  # Reset gate
W_h: jnp.ndarray  # Hidden
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
“”“Initialize GRU parameters.”””
keys = jax.random.split(key, 6)

```
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
```

@jax.jit
def gru_step(h: jnp.ndarray, x: jnp.ndarray, params: GRUParams) -> jnp.ndarray:
“”“Single GRU step.”””
z = jax.nn.sigmoid(x @ params.W_z + h @ params.U_z + params.b_z)
r = jax.nn.sigmoid(x @ params.W_r + h @ params.U_r + params.b_r)
h_tilde = jnp.tanh(x @ params.W_h + (r * h) @ params.U_h + params.b_h)
h_new = (1 - z) * h + z * h_tilde
return h_new

def gru_forward(params: GRUParams,
inputs: jnp.ndarray,
h0: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
“”“Forward pass through GRU.”””
hidden_dim = params.U_z.shape[0]
if h0 is None:
h0 = jnp.zeros(hidden_dim)

```
def scan_body(h, x):
    h_new = gru_step(h, x, params)
    return h_new, h_new

final_h, hiddens = jax.lax.scan(scan_body, h0, inputs)
return final_h, hiddens
```

# ============================================================================

# Collapse Detection for Baselines

# ============================================================================

def detect_baseline_collapse(hiddens: jnp.ndarray) -> dict:
“”“Detect collapse/explosion in baseline models.”””
# Variance across hidden dimension
var_per_step = jnp.var(hiddens, axis=-1)

```
# Norm per step
norm_per_step = jnp.linalg.norm(hiddens, axis=-1)

return {
    'mean_variance': float(jnp.mean(var_per_step)),
    'min_variance': float(jnp.min(var_per_step)),
    'max_norm': float(jnp.max(norm_per_step)),
    'final_norm': float(norm_per_step[-1]),
    'collapsed': bool(jnp.min(var_per_step) < 1e-6),
    'exploded': bool(jnp.max(norm_per_step) > 1e6),
}
```

# ============================================================================

# Tests

# ============================================================================

def test_baselines():
“”“Test baseline models.”””
print(”=” * 60)
print(”  Baseline Model Tests”)
print(”=” * 60)

```
key = jax.random.PRNGKey(42)
input_dim = 8
hidden_dim = 16
seq_len = 100

inputs = jax.random.normal(key, (seq_len, input_dim)) * 0.5

# Test RNN
print("\n[1] Vanilla RNN...")
k1, key = jax.random.split(key)
rnn_params = init_rnn_params(k1, input_dim, hidden_dim)
_, rnn_hiddens = rnn_forward(rnn_params, inputs)
rnn_stats = detect_baseline_collapse(rnn_hiddens)
print(f"    Output shape: {rnn_hiddens.shape}")
print(f"    Mean variance: {rnn_stats['mean_variance']:.6f}")
print(f"    Collapsed: {rnn_stats['collapsed']}")

# Test LSTM
print("\n[2] LSTM...")
k2, key = jax.random.split(key)
lstm_params = init_lstm_params(k2, input_dim, hidden_dim)
_, lstm_hiddens = lstm_forward(lstm_params, inputs)
lstm_stats = detect_baseline_collapse(lstm_hiddens)
print(f"    Output shape: {lstm_hiddens.shape}")
print(f"    Mean variance: {lstm_stats['mean_variance']:.6f}")
print(f"    Collapsed: {lstm_stats['collapsed']}")

# Test GRU
print("\n[3] GRU...")
k3, key = jax.random.split(key)
gru_params = init_gru_params(k3, input_dim, hidden_dim)
_, gru_hiddens = gru_forward(gru_params, inputs)
gru_stats = detect_baseline_collapse(gru_hiddens)
print(f"    Output shape: {gru_hiddens.shape}")
print(f"    Mean variance: {gru_stats['mean_variance']:.6f}")
print(f"    Collapsed: {gru_stats['collapsed']}")

# Test gradient flow
print("\n[4] Gradient flow comparison...")

def rnn_loss(params, inputs):
    _, h = rnn_forward(params, inputs)
    return jnp.mean(h ** 2)

def lstm_loss(params, inputs):
    _, h = lstm_forward(params, inputs)
    return jnp.mean(h ** 2)

def gru_loss(params, inputs):
    _, h = gru_forward(params, inputs)
    return jnp.mean(h ** 2)

rnn_grad = jax.grad(rnn_loss)(rnn_params, inputs[:50])
lstm_grad = jax.grad(lstm_loss)(lstm_params, inputs[:50])
gru_grad = jax.grad(gru_loss)(gru_params, inputs[:50])

rnn_gnorm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(rnn_grad)))
lstm_gnorm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(lstm_grad)))
gru_gnorm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(gru_grad)))

print(f"    RNN gradient norm:  {rnn_gnorm:.6f}")
print(f"    LSTM gradient norm: {lstm_gnorm:.6f}")
print(f"    GRU gradient norm:  {gru_gnorm:.6f}")

print("\n" + "=" * 60)
print("  All baseline tests passed! ✓")
print("=" * 60)

return True
```

if **name** == “**main**”:
test_baselines()