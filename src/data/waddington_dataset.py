import os
import torch
from torch.utils.data import Dataset
import numpy as np

class SyntheticWaddingtonDataset(Dataset):
    """
    Synthetic biological dataset representing a cell moving through a phase transition
    (the Waddington landscape).

    Generates synthetic sequences comprising:
    - y_true: The continuous hidden trajectory.
    - x_raw: A 30-dimensional tensor (20-D continuous sine-wave signals + 10-D sparse signals).
    - mask: A 2-dimensional tensor representing the observability of the two modalities.
    """
    def __init__(self, size: int = 100, seq_len: int = 500):
        """
        Args:
            size (int): Number of sequences in the dataset.
            seq_len (int): Length of each generated sequence.
        """
        self.size = size
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        """
        Generates a single synthetic sequence.

        Returns:
            dict: Containing 'x_raw' (seq_len, 30), 'mask' (seq_len, 2), and 'y_true' (seq_len, 1).
        """
        t = torch.arange(self.seq_len, dtype=torch.float32)
        
        # --- 1. The True State (The Target) ---
        # Center the sigmoid transition randomly between t=200 and t=300
        center = torch.randint(200, 301, (1,)).item()
        steepness = 0.05
        y_true = torch.sigmoid((t - center) * steepness)
        
        # Add a random walk to simulate biological noise
        noise = torch.randn(self.seq_len) * 0.01
        random_walk = torch.cumsum(noise, dim=0)
        y_true = y_true + random_walk
        y_true = y_true.unsqueeze(1)  # [seq_len, 1]
        
        # --- 2. The Modalities (The Raw Data) ---
        # Modality 0: Continuous, 20-Dimensional
        # High-frequency sine wave tracking y_true baseline
        freqs = torch.rand(20) * 0.5 + 0.1
        phases = torch.rand(20) * 2 * np.pi
        time_matrix = t.unsqueeze(1).expand(-1, 20)
        sine_waves = torch.sin(time_matrix * freqs + phases) * 0.2
        mod0 = sine_waves + y_true.expand(-1, 20) + torch.randn(self.seq_len, 20) * 0.05
        
        # Modality 1: Sparse, 10-Dimensional
        # Slow-moving signal tracking y_true with noise
        mod1 = y_true.expand(-1, 10) + torch.randn(self.seq_len, 10) * 0.1
        
        # --- 3. The Masking (Hardware Limits) ---
        mask = torch.zeros(self.seq_len, 2, dtype=torch.float32)
        
        # Modality 0 is fully observed
        mask[:, 0] = 1.0 
        
        # Modality 1 is sparse (e.g., randomly 5% of timesteps)
        sparse_mask = (torch.rand(self.seq_len) < 0.05).float()
        mask[:, 1] = sparse_mask
        
        # CRITICAL: Apply mask to Modality 1. Force unobserved timesteps to exact 0.0
        mod1 = mod1 * sparse_mask.unsqueeze(1)
        
        # Concatenate Modality 0 and Modality 1
        x_raw = torch.cat([mod0, mod1], dim=1)  # [seq_len, 30]
        
        return {
            "x_raw": x_raw,
            "mask": mask,
            "y_true": y_true
        }

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Instantiate dataset and fetch first sample
    dataset = SyntheticWaddingtonDataset(size=1)
    sample = dataset[0]
    x_raw = sample["x_raw"]
    mask = sample["mask"]
    y_true = sample["y_true"]
    
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # 1. 1D y_true trajectory
    axes[0].plot(y_true.numpy(), color='black', linewidth=2)
    axes[0].set_title("y_true Trajectory (Waddington Landscape)")
    axes[0].set_ylabel("Phase Value")
    
    # 2. Heatmap of 30-D x_raw
    im1 = axes[1].imshow(x_raw.numpy().T, aspect='auto', cmap='viridis', interpolation='none')
    axes[1].set_title("x_raw Heatmap (Top 20: Continuous Modality 0, Bottom 10: Sparse Modality 1)")
    axes[1].set_ylabel("Dimension Index")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. 2-D Mask over time
    im2 = axes[2].imshow(mask.numpy().T, aspect='auto', cmap='binary', interpolation='none')
    axes[2].set_title("Mask Observability (Top: Modality 0, Bottom: Modality 1)")
    axes[2].set_xlabel("Time Step")
    axes[2].set_ylabel("Modality Index")
    axes[2].set_yticks([0, 1])
    
    plt.tight_layout()
    
    # Ensure outputs directory exists
    os.makedirs("output/data", exist_ok=True)
    out_path = "output/data/01_synthetic_data_preview.png"
    plt.savefig(out_path)
    print(f"Saved diagnostic preview to {out_path}")
