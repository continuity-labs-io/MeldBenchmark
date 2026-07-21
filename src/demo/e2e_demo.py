import os
import sys

# Setup project root and output directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

output_dir = os.path.join(project_root, 'output')
os.makedirs(output_dir, exist_ok=True)

import torch
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import warnings

warnings.filterwarnings('ignore')

from src.pipeline.aollsm_dataloader import AOLLSMDataset
from torch.utils.data import DataLoader
from src.models.spatial_compressor import SpatialCompressor
from src.models.vector_seq_engine import VectorSeqEngine

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
# We take the max-projection across the Depth axis (dim=0 of the 3D volume)
frame_0_3d = raw_batch[0, 0, 0, :, :, :] 
frame_0_2d_proj, _ = torch.max(frame_0_3d, dim=0) 

# Render the Biological Ground Truth
plt.figure(figsize=(6, 6))
plt.style.use('dark_background')
plt.imshow(frame_0_2d_proj.cpu().numpy(), cmap='magma')
plt.title("Phase 1 Check: 2D Max-Projection (Frame 0)", color='white', fontweight='bold')
plt.axis('off')
plot_path = os.path.join(output_dir, 'plot_1.png')
plt.savefig(plot_path)
print(f'Saved {plot_path}')


print("[*] Injecting Structural Anomaly at Frame 7 (Event Boundary)...")

# Clone the tensor so we don't permanently destroy our ground truth data
experimental_batch = raw_batch.clone()

# At T=7, we inject massive random noise (simulating a cell membrane rupture)
# Multiplying by a huge scalar guarantees the ViT embedding shifts violently
anomaly_noise = torch.randn_like(experimental_batch[:, 7, :, :, :, :]) * experimental_batch.max() * 2.0
experimental_batch[:, 7, :, :, :, :] += anomaly_noise

# Visually verify the anomaly
frame_7_3d = experimental_batch[0, 7, 0, :, :, :]
frame_7_2d_proj, _ = torch.max(frame_7_3d, dim=0)

plt.figure(figsize=(6, 6))
plt.style.use('dark_background')
plt.imshow(frame_7_2d_proj.cpu().numpy(), cmap='magma')
plt.title("Structural Shattering (Frame 7)", color='white', fontweight='bold')
plt.axis('off')
plot_path = os.path.join(output_dir, 'plot_2.png')
plt.savefig(plot_path)
print(f'Saved {plot_path}')


print("[*] Initializing CHRONOS Architecture...")
# Load the pre-trained ViT and the Mamba Engine
compressor = SpatialCompressor().to(device)
mamba_engine = VectorSeqEngine(d_model=768).to(device)

# Set to eval mode to prevent memory exhaustion
compressor.eval()
mamba_engine.eval()

print("[*] Executing Level 1: Spatial Compression (ViT-Base)...")
with torch.no_grad():
    # Flattens the 6D tensor into [Batch, Time, 768]
    latent_sequence = compressor(experimental_batch)
    print(f"    -> Compressed Latent Sequence Shape: {latent_sequence.shape}")

print("[*] Executing Level 2: Continuous Physics Modeling (Mamba-2)...")
with torch.no_grad():
    # Pass through Mamba to predict the next frame and calculate the Cosine Distance
    scalar_loss, dab_metric = mamba_engine(latent_sequence)
    print(f"    -> Continuous Trajectory Processed. Final Loss: {scalar_loss.item():.4f}")

# Extract the DAB metric
dab_scores = dab_metric.cpu().numpy()

# --- THE SCOREBOARD VISUALIZATION ---
plt.figure(figsize=(10, 5))
plt.style.use('dark_background')

# The prediction target ranges from Frame 1 to Frame 9 (Transitions 1 through 9)
time_axis = np.arange(1, len(dab_scores) + 1)

# Plot the stability tracking
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
plot_path = os.path.join(output_dir, 'plot_3.png')
plt.savefig(plot_path)
print(f'Saved {plot_path}')


import pandas as pd
import torch.nn.functional as F

print("[*] Extracting Thermodynamic Metrics (CVI) from Latent Space...")

# 1. Grab the latent sequence generated by the ViT Spatial Compressor
# Shape: [1, 10, 768] -> Detach and squeeze to [10, 768]
z = latent_sequence[0].detach() 
time_steps = z.shape[0]

from src.metrics.thermodynamics import ThermodynamicMetrics

# 2. CALCULATING THE CVI METRIC
metrics = ThermodynamicMetrics(alpha=1000.0, beta=1.0)
cvi_scores = metrics.calculate_cvi(z, window_size=3)

# --- THE THERMODYNAMIC SCOREBOARD ---
plt.figure(figsize=(10, 5))
plt.style.use('dark_background')
time_axis = np.arange(0, len(cvi_scores))

# Plot the CVI tracking
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
plot_path = os.path.join(output_dir, 'plot_4.png')
plt.savefig(plot_path)
print(f'Saved {plot_path}')
