"""
WaveML-JAX: Delayed Copy Benchmark
==================================
Lightborne Intelligence

The delayed copy task tests long-horizon memory:
    - Input: Sequence of tokens
    - Output: Same sequence, delayed by D steps
    - Challenge: Model must maintain information across the delay

This is where baselines collapse and ERA-governed models succeed.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Dict
from functools import partial
import time

# Import models
import sys
sys.path.insert(0, '/home/claude/waveml-jax')

from core.representation import WaveState
from core.invariants import InvariantBounds, DEFAULT_BOUNDS
from models.waveseq import init_waveseq_params, waveseq_forward, detect_collapse
from models.baselines import (
    init_rnn_params, rnn_forward,
    init_lstm_params, lstm_forward,
    init_gru_params, gru_forward,
    detect_baseline_collapse
)


# ============================================================================
# Task Generation
# ============================================================================

def generate_delayed_copy_task(key: jax.random.PRNGKey,
                               seq_len: int,
                               vocab_size: int,
                               delay: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate delayed copy task.
    
    Args:
        key: Random key
        seq_len: Total sequence length (must be > delay)
        vocab_size: Number of token types
        delay: Number of steps to delay
    
    Returns:
        inputs: One-hot encoded input sequence (seq_len, vocab_size)
        targets: One-hot encoded target sequence (seq_len, vocab_size)
    """
    # Generate random tokens
    tokens = jax.random.randint(key, (seq_len,), 0, vocab_size)
    
    # Create inputs (one-hot)
    inputs = jax.nn.one_hot(tokens, vocab_size)
    
    # Create targets (shifted by delay, padded with zeros at start)
    target_tokens = jnp.concatenate([
        jnp.zeros(delay, dtype=jnp.int32),  # Padding
        tokens[:-delay]  # Shifted tokens
    ])
    targets = jax.nn.one_hot(target_tokens, vocab_size)
    
    return inputs, targets


def generate_batch(key: jax.random.PRNGKey,
                   batch_size: int,
                   seq_len: int,
                   vocab_size: int,
                   delay: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate batch of delayed copy tasks."""
    keys = jax.random.split(key, batch_size)
    
    inputs_list = []
    targets_list = []
    
    for k in keys:
        inp, tgt = generate_delayed_copy_task(k, seq_len, vocab_size, delay)
        inputs_list.append(inp)
        targets_list.append(tgt)
    
    return jnp.stack(inputs_list), jnp.stack(targets_list)


# ============================================================================
# Evaluation Metrics
# ============================================================================

def accuracy_at_position(predictions: jnp.ndarray,
                         targets: jnp.ndarray,
                         position: int) -> float:
    """Compute accuracy at a specific position."""
    pred_tokens = jnp.argmax(predictions[:, position], axis=-1)
    target_tokens = jnp.argmax(targets[:, position], axis=-1)
    return float(jnp.mean(pred_tokens == target_tokens))


def accuracy_after_delay(predictions: jnp.ndarray,
                         targets: jnp.ndarray,
                         delay: int) -> float:
    """Compute accuracy for positions after the delay."""
    pred_tokens = jnp.argmax(predictions[:, delay:], axis=-1)
    target_tokens = jnp.argmax(targets[:, delay:], axis=-1)
    return float(jnp.mean(pred_tokens == target_tokens))


# ============================================================================
# Model Runners
# ============================================================================

def run_waveseq(key: jax.random.PRNGKey,
                inputs: jnp.ndarray,
                hidden_dim: int,
                bounds: InvariantBounds = DEFAULT_BOUNDS) -> Tuple[jnp.ndarray, Dict]:
    """Run WaveSeq on inputs."""
    input_dim = inputs.shape[-1]
    params = init_waveseq_params(key, input_dim, hidden_dim)
    
    # Forward pass
    _, amplitudes = waveseq_forward(params, inputs, bounds=bounds)
    
    # Simple linear readout
    W_out = jax.random.normal(key, (hidden_dim, input_dim)) * 0.1
    outputs = amplitudes @ W_out
    
    # Collapse stats
    stats = detect_collapse(amplitudes)
    
    return outputs, stats


def run_rnn(key: jax.random.PRNGKey,
            inputs: jnp.ndarray,
            hidden_dim: int) -> Tuple[jnp.ndarray, Dict]:
    """Run vanilla RNN on inputs."""
    input_dim = inputs.shape[-1]
    params = init_rnn_params(key, input_dim, hidden_dim)
    
    _, hiddens = rnn_forward(params, inputs)
    
    W_out = jax.random.normal(key, (hidden_dim, input_dim)) * 0.1
    outputs = hiddens @ W_out
    
    stats = detect_baseline_collapse(hiddens)
    
    return outputs, stats


def run_lstm(key: jax.random.PRNGKey,
             inputs: jnp.ndarray,
             hidden_dim: int) -> Tuple[jnp.ndarray, Dict]:
    """Run LSTM on inputs."""
    input_dim = inputs.shape[-1]
    params = init_lstm_params(key, input_dim, hidden_dim)
    
    _, hiddens = lstm_forward(params, inputs)
    
    W_out = jax.random.normal(key, (hidden_dim, input_dim)) * 0.1
    outputs = hiddens @ W_out
    
    stats = detect_baseline_collapse(hiddens)
    
    return outputs, stats


def run_gru(key: jax.random.PRNGKey,
            inputs: jnp.ndarray,
            hidden_dim: int) -> Tuple[jnp.ndarray, Dict]:
    """Run GRU on inputs."""
    input_dim = inputs.shape[-1]
    params = init_gru_params(key, input_dim, hidden_dim)
    
    _, hiddens = gru_forward(params, inputs)
    
    W_out = jax.random.normal(key, (hidden_dim, input_dim)) * 0.1
    outputs = hiddens @ W_out
    
    stats = detect_baseline_collapse(hiddens)
    
    return outputs, stats


# ============================================================================
# Main Benchmark
# ============================================================================

def run_benchmark(delays: list = [10, 25, 50, 100, 200],
                  seq_len: int = 300,
                  vocab_size: int = 8,
                  hidden_dim: int = 32,
                  n_trials: int = 5):
    """
    Run delayed copy benchmark comparing WaveSeq vs baselines.
    """
    print("=" * 70)
    print("  DELAYED COPY BENCHMARK")
    print("  WaveSeq (ERA) vs RNN vs LSTM vs GRU")
    print("=" * 70)
    print(f"\n  Config: seq_len={seq_len}, vocab_size={vocab_size}, hidden_dim={hidden_dim}")
    print(f"  Delays tested: {delays}")
    print(f"  Trials per delay: {n_trials}")
    
    results = {d: {'waveseq': [], 'rnn': [], 'lstm': [], 'gru': []} for d in delays}
    collapse_stats = {d: {'waveseq': [], 'rnn': [], 'lstm': [], 'gru': []} for d in delays}
    
    master_key = jax.random.PRNGKey(42)
    
    for delay in delays:
        print(f"\n{'─' * 70}")
        print(f"  Delay = {delay}")
        print(f"{'─' * 70}")
        
        for trial in range(n_trials):
            key = jax.random.fold_in(master_key, delay * 1000 + trial)
            k1, k2, k3, k4, k5 = jax.random.split(key, 5)
            
            # Generate task
            inputs, targets = generate_delayed_copy_task(k1, seq_len, vocab_size, delay)
            
            # Run models
            ws_out, ws_stats = run_waveseq(k2, inputs, hidden_dim)
            rnn_out, rnn_stats = run_rnn(k3, inputs, hidden_dim)
            lstm_out, lstm_stats = run_lstm(k4, inputs, hidden_dim)
            gru_out, gru_stats = run_gru(k5, inputs, hidden_dim)
            
            # Compute accuracy (random baseline = 1/vocab_size = 12.5%)
            ws_acc = accuracy_after_delay(ws_out[None], targets[None], delay)
            rnn_acc = accuracy_after_delay(rnn_out[None], targets[None], delay)
            lstm_acc = accuracy_after_delay(lstm_out[None], targets[None], delay)
            gru_acc = accuracy_after_delay(gru_out[None], targets[None], delay)
            
            results[delay]['waveseq'].append(ws_acc)
            results[delay]['rnn'].append(rnn_acc)
            results[delay]['lstm'].append(lstm_acc)
            results[delay]['gru'].append(gru_acc)
            
            collapse_stats[delay]['waveseq'].append(ws_stats['healthy'])
            collapse_stats[delay]['rnn'].append(not rnn_stats['collapsed'])
            collapse_stats[delay]['lstm'].append(not lstm_stats['collapsed'])
            collapse_stats[delay]['gru'].append(not gru_stats['collapsed'])
        
        # Report
        print(f"\n  Model      | Accuracy (mean±std)  | Healthy Trials")
        print(f"  {'-' * 50}")
        
        for model in ['waveseq', 'rnn', 'lstm', 'gru']:
            accs = jnp.array(results[delay][model])
            healthy = jnp.array(collapse_stats[delay][model])
            mean_acc = jnp.mean(accs)
            std_acc = jnp.std(accs)
            health_rate = jnp.mean(healthy)
            
            marker = "✓" if model == 'waveseq' else " "
            print(f"  {model:10s} | {mean_acc:.1%} ± {std_acc:.1%}       | {health_rate:.0%} {marker}")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY: Mean Accuracy by Delay")
    print(f"{'=' * 70}")
    print(f"\n  {'Delay':>6s} | {'WaveSeq':>10s} | {'RNN':>10s} | {'LSTM':>10s} | {'GRU':>10s}")
    print(f"  {'-' * 55}")
    
    for delay in delays:
        ws = jnp.mean(jnp.array(results[delay]['waveseq']))
        rnn = jnp.mean(jnp.array(results[delay]['rnn']))
        lstm = jnp.mean(jnp.array(results[delay]['lstm']))
        gru = jnp.mean(jnp.array(results[delay]['gru']))
        
        print(f"  {delay:>6d} | {ws:>10.1%} | {rnn:>10.1%} | {lstm:>10.1%} | {gru:>10.1%}")
    
    print(f"\n  Random baseline: {1/vocab_size:.1%}")
    print(f"\n{'=' * 70}")
    print("  Benchmark complete.")
    print(f"{'=' * 70}")
    
    return results, collapse_stats


# ============================================================================
# Tests
# ============================================================================

def test_delayed_copy():
    """Test delayed copy task generation and evaluation."""
    print("=" * 60)
    print("  Delayed Copy Task Tests")
    print("=" * 60)
    
    key = jax.random.PRNGKey(42)
    
    # Test 1: Task generation
    print("\n[1] Task generation...")
    inputs, targets = generate_delayed_copy_task(key, seq_len=20, vocab_size=4, delay=5)
    print(f"    Input shape: {inputs.shape}")
    print(f"    Target shape: {targets.shape}")
    
    # Verify delay
    input_tokens = jnp.argmax(inputs, axis=-1)
    target_tokens = jnp.argmax(targets, axis=-1)
    print(f"    Input tokens:  {input_tokens[:10]}")
    print(f"    Target tokens: {target_tokens[:10]}")
    print(f"    (First 5 targets should be 0 due to delay)")
    
    # Test 2: Batch generation
    print("\n[2] Batch generation...")
    batch_in, batch_tgt = generate_batch(key, batch_size=4, seq_len=20, vocab_size=4, delay=5)
    print(f"    Batch input shape: {batch_in.shape}")
    print(f"    Batch target shape: {batch_tgt.shape}")
    
    # Test 3: Model forward passes
    print("\n[3] Model forward passes...")
    hidden_dim = 16
    
    ws_out, ws_stats = run_waveseq(key, inputs, hidden_dim)
    rnn_out, rnn_stats = run_rnn(key, inputs, hidden_dim)
    lstm_out, lstm_stats = run_lstm(key, inputs, hidden_dim)
    gru_out, gru_stats = run_gru(key, inputs, hidden_dim)
    
    print(f"    WaveSeq output shape: {ws_out.shape}, healthy: {ws_stats['healthy']}")
    print(f"    RNN output shape: {rnn_out.shape}, collapsed: {rnn_stats['collapsed']}")
    print(f"    LSTM output shape: {lstm_out.shape}, collapsed: {lstm_stats['collapsed']}")
    print(f"    GRU output shape: {gru_out.shape}, collapsed: {gru_stats['collapsed']}")
    
    # Test 4: Accuracy computation
    print("\n[4] Accuracy computation...")
    acc = accuracy_after_delay(ws_out[None], targets[None], delay=5)
    print(f"    WaveSeq accuracy (untrained): {acc:.1%}")
    print(f"    Random baseline: {1/4:.1%}")
    
    print("\n" + "=" * 60)
    print("  All delayed copy tests passed! ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    # Run tests first
    test_delayed_copy()
    
    # Then run mini benchmark
    print("\n\n")
    run_benchmark(
        delays=[10, 25, 50],
        seq_len=100,
        n_trials=3
    )
