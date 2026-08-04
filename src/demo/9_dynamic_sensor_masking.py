import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None

from src.pipeline.neocortical_assembloid_dataloader import NeocorticalAssembloidDataset
from src.utils.device import get_optimal_device

class DynamicMaskingEngine(nn.Module):
    """
    A Continuous-Time State Space Engine that dynamically routes around 
    hardware sensor failures (NaNs) in real-time.
    """
    def __init__(self, input_dim=114, d_model=256, d_state=64):
        super().__init__()
        self.input_dim = input_dim
        
        # THE MOAT: The input dimension is doubled (114 data + 114 mask)
        # This explicit bottleneck forces the surviving sensors to compensate.
        self.mask_encoder = nn.Sequential(
            nn.Linear(input_dim * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Mamba-2 Backbone
        if Mamba2 is not None:
            self.mamba_block1 = Mamba2(d_model=d_model, d_state=d_state)
            self.mamba_block2 = Mamba2(d_model=d_model, d_state=d_state)
        else:
            self.mamba_block1 = nn.Identity()
            self.mamba_block2 = nn.Identity()
        
        # Forward Predictor
        self.predictor = nn.Linear(d_model, input_dim)

    def forward(self, x):
        """
        x: [Batch, Time, Features] containing NaNs where hardware dropped.
        """
        # 1. Detect missing sensors (1.0 if dead/NaN, 0.0 if healthy)
        mask = torch.isnan(x).float()
        
        # 2. Sanitize the input (Zero-fill NaNs so PyTorch math doesn't explode)
        x_safe = torch.nan_to_num(x, nan=0.0)
        
        # 3. The Crucial Step: Concatenate Data + Mask 
        # Shape becomes [Batch, Time, input_dim * 2]
        x_combined = torch.cat([x_safe, mask], dim=-1)
        
        # 4. Project into the continuous latent space
        h = self.mask_encoder(x_combined)
        
        # 5. Continuous-time sequence modeling
        h = self.mamba_block1(h)
        h = self.mamba_block2(h)
        
        # 6. Predict the true biological state (even the missing parts)
        preds = self.predictor(h)
        
        return preds, mask


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    device = get_optimal_device(allow_mps=False)
    print(f"\n[*] Booting BDC-RFC-004 Sensor Masking Demo on: {device.type.upper()}")
    
    INPUT_DIM = 114
    dataset = NeocorticalAssembloidDataset(time_steps=200, num_channels=256, latent_dim=INPUT_DIM)
    dataset_iter = iter(dataset)
    
    engine = DynamicMaskingEngine(input_dim=INPUT_DIM, d_model=256).to(device)
    optimizer = optim.AdamW(engine.parameters(), lr=1e-3)
    
    # --- 1. BURN-IN TRAINING (Learning the Spatial Covariance) ---
    print("[*] Training network to learn multi-modal covariance (Burn-In)...")
    engine.train()
    
    for i in range(25):
        seq, _ = next(dataset_iter)
        seq = seq.unsqueeze(0).to(device) # Clean ground truth [1, 200, 114]
        
        # Simulate training dropouts (randomly drop 10% of sensors to teach the encoder)
        dropout_mask = (torch.rand_like(seq) > 0.1)
        seq_corrupt = torch.where(dropout_mask, seq, torch.tensor(float('nan'), device=device))
        
        optimizer.zero_grad()
        
        # Forward pass predicting T+1
        pred_t_plus_1, _ = engine(seq_corrupt[:, :-1, :])
        target_t_plus_1 = seq[:, 1:, :] # Predict against the UNCORRUPTED ground truth
        
        loss = F.mse_loss(pred_t_plus_1, target_t_plus_1)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 5 == 0:
            print(f"    Iteration {i+1}/25 | Total Loss: {loss.item():.4f}")

    # --- 2. THE WET-LAB DISASTER SIMULATION ---
    print("\n[*] Simulating catastrophic sensor failure (Voltage Array disconnected)...")
    engine.eval()
    
    true_seq, _ = next(dataset_iter)
    true_seq = true_seq.unsqueeze(0).to(device)
    
    test_seq = true_seq.clone()
    
    # At T=100, the last two features (Omega_VoltRed, Omega_VoltGrn) completely drop out to NaN
    DROP_FRAME = 100
    test_seq[:, DROP_FRAME:, 112:] = float('nan') 
    
    with torch.no_grad():
        # Masker intercepts the NaNs and infers the voltage based on the optical shape (Features 0-111)
        pred_seq, valid_mask = engine(test_seq[:, :-1, :])
        
        # Calculate the error between the Masker's guess and the ground truth we hid
        imputed_voltage = pred_seq[0, DROP_FRAME-1:, 112:].cpu().numpy()
        true_voltage = true_seq[0, DROP_FRAME:, 112:].cpu().numpy()
        
    print("[*] Generating Dashboard...")
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Panel 1: What the Mamba Engine Received
    im1 = ax1.imshow(test_seq[0].T.cpu().numpy(), aspect='auto', cmap='viridis', origin='lower')
    ax1.axvline(x=DROP_FRAME, color='red', linestyle='--', linewidth=2, label='Sensors Dropped (NaN)')
    ax1.set_title("114-D Tensor with Catastrophic NaN Dropout", color='white', fontweight='bold')
    ax1.set_ylabel("Features")
    ax1.legend()
    fig.colorbar(im1, ax=ax1)
    
    # Panel 2: The Imputation Accuracy
    t_drop = np.arange(DROP_FRAME, 200)
    ax2.plot(t_drop, true_voltage[:, 0], color='cyan', label='Ground Truth Voltage', linewidth=2)
    ax2.plot(t_drop, imputed_voltage[:, 0], color='orange', linestyle='--', label='Masker Imputation', linewidth=2)
    ax2.axvline(x=DROP_FRAME, color='red', linestyle='--', linewidth=2)
    ax2.set_title("Continuous-Time Spatial Imputation (Omega Voltage Track)", color='white', fontweight='bold')
    ax2.set_ylabel("Amplitude")
    ax2.set_xlabel("Time Step")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "9_sensor_dropout_demo.png")
    plt.savefig(output_path, dpi=300)
    print(f"[+] Demo Complete. Dashboard saved to: {output_path}")

if __name__ == "__main__":
    main()
