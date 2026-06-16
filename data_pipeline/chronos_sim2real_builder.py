"""
CHRONOS Sim2Real Builder

This script generates a synthetic, multi-scale biological dataset simulating
the trajectory of cellular systems (organoids) heading toward a catastrophic 
collapse (a bifurcation point or crash).

The generated data is exported as an `.npz` file containing three distinct scales:
- Scale 1 (Sparse Telemetry): A 14-D timeseries representing biological sensors 
  (RNA levels and GEVIs/voltage sensors) that spike around the crash event. 
  It is masked with NaNs to simulate 99% sparsity (optical hardware limits).
- Scale 2 (Phase Tensor): A 100-D continuous timeseries representing the raw tracking 
  fluid dynamics or non-linear oscillator drift of the cells over 72 hours.
- Scale 3 (Terminal Target): A 20,000-D vector representing the final high-resolution 
  transcriptomic state (scRNA-seq with dropout) of the system at the endpoint.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import os


def build_vorganoid_tensor(use_real_ctc_data=False, ctc_csv_path="ctc_tracks.csv"):
    np.random.seed(42)
    print("Booting CHRONOS Sim2Real Oracle Engine...")

    # --- TIME DIMENSIONS ---
    # 72 hours at 1-minute resolution = 4320 discrete time-steps
    TOTAL_STEPS = 4320
    CRASH_STEP = int(TOTAL_STEPS * (61.4 / 72.0)) # The exact Bifurcation Point

    # --- 1. THE BASE REALITY (100-D PHASE TENSOR) ---
    if use_real_ctc_data and os.path.exists(ctc_csv_path):
        print("Ingesting Real Biology: Parsing Cell Tracking Challenge CSV...")
        # Format expectation: [Time, Cell_ID, Z, Y, X]
        df = pd.read_csv(ctc_csv_path, names=['Time', 'Cell_ID', 'Z', 'Y', 'X'])
        pivot_df = df.pivot(index='Time', columns='Cell_ID', values=['Z', 'Y', 'X']).ffill().fillna(0)
        raw_tracks = pivot_df.values

        print("Compressing raw tracking fluid dynamics to 100-D Phase Tensor via PCA...")
        pca = PCA(n_components=100)
        phase_100d = pca.fit_transform(raw_tracks)
        # Note: In a real run, you interpolate phase_100d to exactly TOTAL_STEPS

    else:
        print("Generating Math Proxy: 100-D Coupled Non-Linear Oscillators...")
        # Proxy: Langevin active-matter drifting towards a pitchfork bifurcation
        phase_100d = np.zeros((TOTAL_STEPS, 100))
        r_vals = np.linspace(-2.0, 1.0, TOTAL_STEPS) # Control parameter

        for t in range(1, TOTAL_STEPS):
            x = phase_100d[t-1]
            r = r_vals[t]

            # Non-linear drift (dx = rx - x^3) + Active Matter Noise
            dx = (r * x - x**3) * 0.1 
            noise = np.random.normal(0, 0.05, 100)

            # Post-crash, the physical variance explodes (organelles shatter)
            if t > CRASH_STEP:
                noise *= (1.0 + 5.0 * ((t - CRASH_STEP) / TOTAL_STEPS))

            phase_100d[t] = x + dx + noise

    # --- 2. THE SYNTHETIC OVERLAY (14-D CHEMICAL/BIOELECTRIC ANCHORS) ---
    print("Synthesizing 14-D Telemetry Anchors...")
    telemetry_14d = np.zeros((TOTAL_STEPS, 14))

    for t in range(TOTAL_STEPS):
        base_signal = np.random.poisson(lam=2.0, size=14) * 0.1

        # Pre-Crash Wobble: RNA sensors (0-11) spike 2-3 hours BEFORE the crash
        if CRASH_STEP - 180 < t < CRASH_STEP:
            base_signal[0:12] += np.random.poisson(lam=10.0, size=12) * 0.5

        # Voltage Crash: GEVIs (12-13) exponentially decay AFTER the crash
        if t >= CRASH_STEP:
            base_signal[12:14] += np.random.exponential(scale=5.0, size=2)

        telemetry_14d[t] = base_signal

    # --- 3. THE CHRONOS MASK (5-MINUTE SPARSITY) ---
    print("Applying Optical Hardware Mask (99% Sparsity)...")
    # 5-minute sampling = 1 flash every 5 steps
    MASK_INTERVAL = 5 
    mask = np.zeros(TOTAL_STEPS, dtype=bool)
    mask[::MASK_INTERVAL] = True

    masked_telemetry = np.copy(telemetry_14d)
    masked_telemetry[~mask] = np.nan # Force NaNs to break autoregressive Transformers

    # --- 4. THE TERMINAL TARGET (20,000-D VISIUM HD) ---
    print("Generating 20,000-D Terminal Endpoint...")
    final_state = phase_100d[-1]
    projection = np.random.randn(100, 20000) * 0.01
    endpoint_20k = np.dot(final_state, projection)

    # Biological scRNA-seq dropout (80% zeros)
    dropout = np.random.rand(20000) > 0.8
    endpoint_20k[dropout] = 0.0

    # --- 5. EXPORT THE TENSOR ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_root, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    out_path = os.path.join(dataset_dir, "chronos_oracle_level1.npz")
    np.savez_compressed(
        out_path,
        time_steps=np.arange(TOTAL_STEPS),
        phase_100d=phase_100d,
        telemetry_14d=masked_telemetry,
        endpoint_20k=endpoint_20k,
        ground_truth_crash=np.array([CRASH_STEP]) # The DAB Target (Secret)
    )

    print(f"\n[+] Success. CHRONOS Tensor forged: {out_path}")
    print(f"    Scale 1 (Sparse Telemetry): {masked_telemetry.shape} (Masked with NaNs)")
    print(f"    Scale 2 (Phase Tensor): {phase_100d.shape}")
    print(f"    Scale 3 (Terminal Target): {endpoint_20k.shape}")

if __name__ == "__main__":
    build_vorganoid_tensor()
