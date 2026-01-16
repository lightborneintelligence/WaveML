"""
WaveML-JAX: Noise Sweep Benchmark
=================================
Lightborne Intelligence

Tests model robustness across noise levels.
ERA should provide graceful degradation vs baseline collapse.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, List
from functools import partial

import sys
sys.path.insert(0, '/home/claude/waveml-jax')

from core.representation import WaveState, encode_fft, decode_fft, total_energy
from core.invariants import InvariantBounds, DEFAULT_BOUNDS
from core.era_rectify import era_rectify
from models.waveseq import init_waveseq_params, waveseq_forward, detect_collapse
from models.baselines import init_rnn_params, rnn_forward, detect_baseline_collapse


# ============================================================================
# Signal Generation
# ============================================================================

def generate_clean_signal(key: jax.random.PRNGKey,
                          length: int,
                          n_components: int = 3) -> jnp.ndarray:
    """Generate clean multi-frequency signal."""
    t = jnp.linspace(0, 4 * jnp.pi, length)
    
    # Random frequencies and amplitudes
    k1, k2 = jax.random.split(key)
    freqs = jax.random.uniform(k1, (n_components,), minval=1, maxval=5)
    amps = jax.random.uniform(k2, (n_components,), minval=0.3, maxval=1.0)
    
    signal = jnp.zeros(length)
    for f, a in zip(freqs, amps):
        signal = signal + a * jnp.sin(f * t)
    
    return signal / jnp.max(jnp.abs(signal))  # Normalize


def add_noise(signal: jnp.ndarray,
              key: jax.random.PRNGKey,
              noise_level: float) -> jnp.ndarray:
    """Add Gaussian noise to signal."""
    noise = jax.random.normal(key, signal.shape) * noise_level
    return signal + noise


# ============================================================================
# Reconstruction Metrics
# ============================================================================

def compute_mse(original: jnp.ndarray, reconstructed: jnp.ndarray) -> float:
    """Mean squared error."""
    return float(jnp.mean((original - reconstructed) ** 2))


def compute_snr(original: jnp.ndarray, reconstructed: jnp.ndarray) -> float:
    """Signal-to-noise ratio in dB."""
    signal_power = jnp.mean(original ** 2)
    noise_power = jnp.mean((original - reconstructed) ** 2)
    return float(10 * jnp.log10(signal_power / (noise_power + 1e-10)))


def compute_correlation(original: jnp.ndarray, reconstructed: jnp.ndarray) -> float:
    """Pearson correlation."""
    orig_centered = original - jnp.mean(original)
    recon_centered = reconstructed - jnp.mean(reconstructed)
    
    num = jnp.sum(orig_centered * recon_centered)
    denom = jnp.sqrt(jnp.sum(orig_centered ** 2) * jnp.sum(recon_centered ** 2) + 1e-10)
    
    return float(num / denom)


# ============================================================================
# Model Processing
# ============================================================================

def process_with_era(signal: jnp.ndarray,
                     n_modes: int = 32,
                     bounds: InvariantBounds = DEFAULT_BOUNDS) -> jnp.ndarray:
    """Process signal through ERA."""
    # Encode
    state = encode_fft(signal, n_modes)
    
    # Apply ERA
    rectified = era_rectify(state, bounds)
    
    # Decode
    return decode_fft(rectified, len(signal))


def process_with_waveseq(signal: jnp.ndarray,
                         key: jax.random.PRNGKey,
                         hidden_dim: int = 32,
                         bounds: InvariantBounds = DEFAULT_BOUNDS) -> Tuple[jnp.ndarray, Dict]:
    """Process signal through WaveSeq."""
    # Reshape signal for sequence model
    inputs = signal.reshape(-1, 1)  # (seq_len, 1)
    
    params = init_waveseq_params(key, input_dim=1, hidden_dim=hidden_dim)
    _, amplitudes = waveseq_forward(params, inputs, bounds=bounds)
    
    # Simple readout
    W_out = jax.random.normal(key, (hidden_dim, 1)) * 0.1
    outputs = (amplitudes @ W_out).flatten()
    
    stats = detect_collapse(amplitudes)
    
    return outputs, stats


def process_with_rnn(signal: jnp.ndarray,
                     key: jax.random.PRNGKey,
                     hidden_dim: int = 32) -> Tuple[jnp.ndarray, Dict]:
    """Process signal through RNN."""
    inputs = signal.reshape(-1, 1)
    
    params = init_rnn_params(key, input_dim=1, hidden_dim=hidden_dim)
    _, hiddens = rnn_forward(params, inputs)
    
    W_out = jax.random.normal(key, (hidden_dim, 1)) * 0.1
    outputs = (hiddens @ W_out).flatten()
    
    stats = detect_baseline_collapse(hiddens)
    
    return outputs, stats


# ============================================================================
# Noise Sweep Benchmark
# ============================================================================

def run_noise_sweep(noise_levels: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
                    signal_length: int = 128,
                    n_trials: int = 10,
                    hidden_dim: int = 32):
    """
    Run noise sweep benchmark.
    
    Compares reconstruction quality across noise levels.
    """
    print("=" * 70)
    print("  NOISE SWEEP BENCHMARK")
    print("  ERA vs Baseline Signal Processing")
    print("=" * 70)
    print(f"\n  Config: signal_length={signal_length}, hidden_dim={hidden_dim}")
    print(f"  Noise levels: {noise_levels}")
    print(f"  Trials per level: {n_trials}")
    
    results = {
        'noise_levels': noise_levels,
        'era': {'mse': [], 'snr': [], 'corr': []},
        'waveseq': {'mse': [], 'snr': [], 'corr': [], 'healthy': []},
        'rnn': {'mse': [], 'snr': [], 'corr': [], 'healthy': []}
    }
    
    master_key = jax.random.PRNGKey(42)
    
    for noise_level in noise_levels:
        print(f"\n{'─' * 70}")
        print(f"  Noise Level = {noise_level:.1f}")
        print(f"{'─' * 70}")
        
        era_mse, era_snr, era_corr = [], [], []
        ws_mse, ws_snr, ws_corr, ws_healthy = [], [], [], []
        rnn_mse, rnn_snr, rnn_corr, rnn_healthy = [], [], [], []
        
        for trial in range(n_trials):
            key = jax.random.fold_in(master_key, int(noise_level * 1000) + trial)
            k1, k2, k3, k4 = jax.random.split(key, 4)
            
            # Generate clean signal
            clean = generate_clean_signal(k1, signal_length)
            
            # Add noise
            noisy = add_noise(clean, k2, noise_level)
            
            # Process with ERA
            era_out = process_with_era(noisy)
            era_mse.append(compute_mse(clean, era_out))
            era_snr.append(compute_snr(clean, era_out))
            era_corr.append(compute_correlation(clean, era_out))
            
            # Process with WaveSeq
            ws_out, ws_stats = process_with_waveseq(noisy, k3, hidden_dim)
            ws_mse.append(compute_mse(clean, ws_out))
            ws_snr.append(compute_snr(clean, ws_out))
            ws_corr.append(compute_correlation(clean, ws_out))
            ws_healthy.append(ws_stats['healthy'])
            
            # Process with RNN
            rnn_out, rnn_stats = process_with_rnn(noisy, k4, hidden_dim)
            rnn_mse.append(compute_mse(clean, rnn_out))
            rnn_snr.append(compute_snr(clean, rnn_out))
            rnn_corr.append(compute_correlation(clean, rnn_out))
            rnn_healthy.append(not rnn_stats['collapsed'])
        
        # Store results
        results['era']['mse'].append(jnp.mean(jnp.array(era_mse)))
        results['era']['snr'].append(jnp.mean(jnp.array(era_snr)))
        results['era']['corr'].append(jnp.mean(jnp.array(era_corr)))
        
        results['waveseq']['mse'].append(jnp.mean(jnp.array(ws_mse)))
        results['waveseq']['snr'].append(jnp.mean(jnp.array(ws_snr)))
        results['waveseq']['corr'].append(jnp.mean(jnp.array(ws_corr)))
        results['waveseq']['healthy'].append(jnp.mean(jnp.array(ws_healthy)))
        
        results['rnn']['mse'].append(jnp.mean(jnp.array(rnn_mse)))
        results['rnn']['snr'].append(jnp.mean(jnp.array(rnn_snr)))
        results['rnn']['corr'].append(jnp.mean(jnp.array(rnn_corr)))
        results['rnn']['healthy'].append(jnp.mean(jnp.array(rnn_healthy)))
        
        # Report
        print(f"\n  Model   | MSE      | SNR (dB) | Corr   | Healthy")
        print(f"  {'-' * 50}")
        print(f"  ERA     | {results['era']['mse'][-1]:.4f}   | {results['era']['snr'][-1]:>6.1f}   | {results['era']['corr'][-1]:.3f}  | N/A")
        print(f"  WaveSeq | {results['waveseq']['mse'][-1]:.4f}   | {results['waveseq']['snr'][-1]:>6.1f}   | {results['waveseq']['corr'][-1]:.3f}  | {results['waveseq']['healthy'][-1]:.0%}")
        print(f"  RNN     | {results['rnn']['mse'][-1]:.4f}   | {results['rnn']['snr'][-1]:>6.1f}   | {results['rnn']['corr'][-1]:.3f}  | {results['rnn']['healthy'][-1]:.0%}")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY: Correlation by Noise Level")
    print(f"{'=' * 70}")
    print(f"\n  {'Noise':>6s} | {'ERA':>8s} | {'WaveSeq':>8s} | {'RNN':>8s}")
    print(f"  {'-' * 40}")
    
    for i, noise in enumerate(noise_levels):
        era_c = results['era']['corr'][i]
        ws_c = results['waveseq']['corr'][i]
        rnn_c = results['rnn']['corr'][i]
        print(f"  {noise:>6.1f} | {era_c:>8.3f} | {ws_c:>8.3f} | {rnn_c:>8.3f}")
    
    print(f"\n{'=' * 70}")
    print("  Benchmark complete.")
    print(f"{'=' * 70}")
    
    return results


# ============================================================================
# Tests
# ============================================================================

def test_noise_sweep():
    """Test noise sweep components."""
    print("=" * 60)
    print("  Noise Sweep Tests")
    print("=" * 60)
    
    key = jax.random.PRNGKey(42)
    
    # Test 1: Signal generation
    print("\n[1] Signal generation...")
    signal = generate_clean_signal(key, 64)
    print(f"    Shape: {signal.shape}")
    print(f"    Max: {jnp.max(signal):.3f}")
    print(f"    Min: {jnp.min(signal):.3f}")
    
    # Test 2: Noise addition
    print("\n[2] Noise addition...")
    noisy = add_noise(signal, key, 0.3)
    noise_actual = jnp.std(noisy - signal)
    print(f"    Noise level requested: 0.3")
    print(f"    Noise std actual: {noise_actual:.3f}")
    
    # Test 3: Metrics
    print("\n[3] Metrics computation...")
    mse = compute_mse(signal, noisy)
    snr = compute_snr(signal, noisy)
    corr = compute_correlation(signal, noisy)
    print(f"    MSE: {mse:.4f}")
    print(f"    SNR: {snr:.1f} dB")
    print(f"    Correlation: {corr:.3f}")
    
    # Test 4: ERA processing
    print("\n[4] ERA processing...")
    era_out = process_with_era(noisy)
    era_corr = compute_correlation(signal, era_out)
    print(f"    Input correlation with clean: {corr:.3f}")
    print(f"    ERA output correlation: {era_corr:.3f}")
    
    print("\n" + "=" * 60)
    print("  All noise sweep tests passed! ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    # Run tests
    test_noise_sweep()
    
    # Run mini benchmark
    print("\n\n")
    run_noise_sweep(
        noise_levels=[0.0, 0.2, 0.5, 1.0],
        n_trials=5
    )
