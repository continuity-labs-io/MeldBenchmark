import os
import sys
import warnings

# Setup project root and output directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

output_dir = os.path.join(project_root, 'output')
os.makedirs(output_dir, exist_ok=True)

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F

from src.pipeline.aollsm_dataloader import AOLLSMDataset
from torch.utils.data import DataLoader
from src.models.spatial_compressor import SpatialCompressor
from src.models.vector_seq_engine import VectorSeqEngine
from src.metrics.thermodynamics import ThermodynamicMetrics

warnings.filterwarnings('ignore')

def plot_frame_projection(frame_3d, title, filename):
    """Takes a 3D frame, computes 2D max-projection, and saves the plot."""
    frame_2d_proj, _ = torch.max(frame_3d, dim=0) 
    
    plt.figure(figsize=(6, 6))
    plt.style.use('dark_background')
    plt.imshow(frame_2d_proj.cpu().numpy(), cmap='magma')
    plt.title(title, color='white', fontweight='bold')
    plt.axis('off')
    
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path)
    print(f'Saved {plot_path}')
    plt.close()

def plot_dab_metric(dab_scores, filename):
    """Plots the Distance-to-Absorbing-Boundary (DAB) metric."""
    plt.figure(figsize=(10, 5))
    plt.style.use('dark_background')
    time_axis = np.arange(1, len(dab_scores) + 1)
    
    plt.plot(time_axis, dab_scores, marker='o', color="cyan", linewidth=3, markersize=8, label="Prediction Error (DAB)")
    
    # Highlight the anomaly zone (Predicting Frame 7 happens at Transition step 7)
    plt.axvline(x=7, color='crimson', linestyle='--', linewidth=2, label='Injected Anomaly Target (T6->T7)')
    plt.axvspan(6.5, 7.5, color='crimson', alpha=0.15)
    
    plt.title("Distance-to-Absorbing-Boundary (DAB) Metric vs Time", fontsize=14, fontweight='bold', color='white')
    plt.xlabel("Temporal Transition Step", fontsize=12, color='white')
    plt.ylabel("Surprise / Cosine Distance", fontsize=12, color='white')
    plt.xticks(time_axis, [f"T{i-1}->T{i}" for i in time_axis])
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path)
    print(f'Saved {plot_path}')
    plt.close()

def plot_cvi_metric(cvi_scores, filename):
    """Plots the Critical Variance Index (CVI) metric."""
    plt.figure(figsize=(10, 5))
    plt.style.use('dark_background')
    time_axis = np.arange(0, len(cvi_scores))
    
    plt.plot(time_axis, cvi_scores, marker='o', color="magenta", linewidth=3, markersize=8, label="Critical Variance Index (CVI)")
    
    # Highlight the anomaly zone (The noise was injected at Frame 7)
    plt.axvline(x=7, color='white', linestyle=':', linewidth=2, label='Structural Tipping Point (T=7)')
    plt.axvspan(6.5, 7.5, color='crimson', alpha=0.15)
    
    plt.title("Thermodynamic Stability: Critical Variance Index (CVI)", fontsize=14, fontweight='bold', color='white')
    plt.xlabel("Time Step (Frames)", fontsize=12, color='white')
    plt.ylabel("CVI Amplitude (Variance + AR1)", fontsize=12, color='white')
    plt.xticks(time_axis, [f"T{i}" for i in time_axis])
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path)
    print(f'Saved {plot_path}')
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Booting MELD End-to-End Compiler on: {device.type.upper()}")

    print("[*] Ingesting AO-LLSM Optical Telemetry...")
    data_dir = os.path.join(project_root, "dataset/raw_tiffs")
    SEQUENCE_LENGTH = 10 

    # Initialize the dataset
    dataset = AOLLSMDataset(data_dir=data_dir, num_frames=SEQUENCE_LENGTH, crop_size=(128, 128, 128))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Grab the first batch
    raw_batch = next(iter(dataloader)).to(device)
    print(f"[*] Loaded Raw Tensor Shape: {raw_batch.shape} -> [Batch, Time, Channel, Depth, Height, Width]")

    # Extract Frame 0, Channel 0 for visual inspection
    frame_0_3d = raw_batch[0, 0, 0, :, :, :] 
    plot_frame_projection(frame_0_3d, "Phase 1 Check: 2D Max-Projection (Frame 0)", 'plot_1.png')

    print("[*] Injecting Structural Anomaly at Frame 7 (Event Boundary)...")
    experimental_batch = raw_batch.clone()

    # At T=7, we inject massive random noise (simulating a cell membrane rupture)
    anomaly_noise = torch.randn_like(experimental_batch[:, 7, :, :, :, :]) * experimental_batch.max() * 2.0
    experimental_batch[:, 7, :, :, :, :] += anomaly_noise

    frame_7_3d = experimental_batch[0, 7, 0, :, :, :]
    plot_frame_projection(frame_7_3d, "Structural Shattering (Frame 7)", 'plot_2.png')

    print("[*] Initializing CHRONOS Architecture...")
    compressor = SpatialCompressor().to(device)
    mamba_engine = VectorSeqEngine(d_model=768).to(device)
    compressor.eval()
    mamba_engine.eval()

    print("[*] Executing Level 1: Spatial Compression (ViT-Base)...")
    with torch.no_grad():
        latent_anomalous = compressor(experimental_batch)
        latent_healthy = compressor(raw_batch)
        print(f"    -> Compressed Latent Sequence Shape: {latent_anomalous.shape}")

    print("[*] Executing Level 2: Continuous Physics Modeling (Mamba-2)...")
    with torch.no_grad():
        scalar_loss, _ = mamba_engine(latent_anomalous)
        print(f"    -> Continuous Trajectory Processed. Final Loss: {scalar_loss.item():.4f}")

    from src.metrics.thermodynamics import ThermodynamicMetrics
    print("[*] Extracting Thermodynamic Metrics (CVI, DAB, Hysteresis)...")

    z_anomalous = latent_anomalous[0].detach() 
    z_healthy = latent_healthy[0].detach()
    time_steps = z_anomalous.shape[0]

    metrics = ThermodynamicMetrics(alpha=500.0, beta=1.0)
    cvi_scores = metrics.calculate_cvi(z_anomalous, window_size=3)
    dab_scores = metrics.calculate_dab(z_anomalous, window_size=4)

    print("[*] Simulating Biological Rescue & Calculating Hysteresis...")
    # Take the shattered latent state (T=7) and let Mamba autoregressively predict recovery
    z_shattered = latent_anomalous[:, 7:8, :] 
    rescue_trajectory = [z_shattered.squeeze(0)]

    z_curr = z_shattered
    with torch.no_grad():
        for _ in range(2): # Predict T=8 and T=9 recovery steps
            h_state = mamba_engine.mamba(z_curr)
            z_next = mamba_engine.proj(h_state[:, -1:, :])
            rescue_trajectory.append(z_next.squeeze(0))
            z_curr = torch.cat([z_curr, z_next], dim=1)

    z_rescue_path = torch.cat(rescue_trajectory, dim=0) # [3, 768]
    z_healthy_path = z_healthy[7:10, :] # The original healthy trajectory [3, 768]

    hysteresis_scalar, divergence_curve = metrics.calculate_hysteresis(z_healthy_path, z_rescue_path)
    print(f"    -> Thermodynamic Hysteresis (Scar Area): {hysteresis_scalar:.4f}")

    # ==========================================
    # THE THERMODYNAMIC SCOREBOARDS
    # ==========================================
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    plt.style.use('dark_background')
    time_axis = np.arange(time_steps)

    # Plot 1: DAB
    ax1.plot(time_axis, dab_scores, marker='o', color="cyan", linewidth=3, label="Jacobian Stability (DAB)")
    ax1.axvline(x=7, color='crimson', linestyle='--', linewidth=2)
    ax1.axvspan(6.5, 7.5, color='crimson', alpha=0.15)
    ax1.set_title("True DAB (Dynamic Mode Decomposition Eigenvalues)", color='white', fontweight='bold')
    ax1.set_ylabel("Distance to Singularity")
    ax1.set_xticks(time_axis)
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # Plot 2: CVI
    ax2.plot(time_axis, cvi_scores, marker='s', color="magenta", linewidth=3, label="Critical Variance Index (CVI)")
    ax2.axvline(x=7, color='white', linestyle=':', linewidth=2)
    ax2.axvspan(6.5, 7.5, color='crimson', alpha=0.15)
    ax2.set_title("Structural Wobble (CVI)", color='white', fontweight='bold')
    ax2.set_ylabel("CVI Amplitude")
    ax2.set_xticks(time_axis)
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    # Plot 3: Hysteresis
    t_rescue = np.arange(len(divergence_curve))
    ax3.plot(t_rescue, divergence_curve, marker='^', color="springgreen", linewidth=3, label="Path Divergence")
    ax3.fill_between(t_rescue, divergence_curve, color="springgreen", alpha=0.2, label=f"Hysteresis Area: {hysteresis_scalar:.2f}")
    ax3.set_title("Morphological Hysteresis (Rescue vs. Healthy Trajectory)", color='white', fontweight='bold')
    ax3.set_xlabel("Recovery Steps", color='white')
    ax3.set_ylabel("Latent Distance (L2 Norm)")
    ax3.set_xticks(t_rescue)
    ax3.set_xticklabels([f"T{i+7}" for i in t_rescue])
    ax3.legend()
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'plot_metrics.png')
    plt.savefig(plot_path)
    print(f'Saved {plot_path}')

if __name__ == "__main__":
    main()
