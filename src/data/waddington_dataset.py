import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import math

class SyntheticWaddingtonDataset(Dataset):
    """
    Synthetic biological dataset representing a cell moving through a phase transition
    (the Waddington landscape).

    Generates synthetic sequences comprising:
    - y_true: The 1D target tracking discrete phase transitions.
    - x_raw: A 30-dimensional tensor composed of two modalities:
      - Modality 0 (20D): Continuous background noise (Gaussian + sine waves) with no
        causal link to the target (prevents shortcut learning).
      - Modality 1 (10D): Sparse causal driver that tracks y_true but is only ~5% active.
    - mask: A 2-dimensional tensor representing the observability of the two modalities.
    """
    def __init__(self, size=100, seq_len=500):
        self.size = size
        self.seq_len = seq_len
        self.W_1 = torch.randn(1, 10)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # The Target (y_true)
        y_true = torch.zeros(self.seq_len, 1)
        jump1 = torch.randint(100, 200, (1,)).item()
        jump2 = torch.randint(300, 400, (1,)).item()
        y_true[jump1:jump2] = 1.0
        y_true[jump2:] = 2.0
        y_true += torch.randn(self.seq_len, 1) * 0.02

        # Modality 0 (Continuous Voltage, 20D) -> Pure Background Noise
        # Generate pure Gaussian noise and random sine waves
        t = torch.arange(self.seq_len).float().unsqueeze(1).expand(-1, 20)
        freqs = torch.rand(20) * 0.5 + 0.1
        phases = torch.rand(20) * 2 * math.pi
        sine_waves = torch.sin(t * freqs + phases) * 0.2
        modality_0 = sine_waves + torch.randn(self.seq_len, 20) * 0.05

        # Modality 1 (Sparse Epigenetics, 10D) -> The Causal Driver
        modality_1 = y_true * self.W_1 + torch.randn(self.seq_len, 10) * 0.05

        # The Mask
        mask_0 = torch.ones(self.seq_len, 1)
        mask_1 = (torch.rand(self.seq_len, 1) > 0.95).float()
        
        # Hack to ensure observability
        if jump1 + 5 < self.seq_len:
            mask_1[jump1 + 5] = 1.0
        if jump2 + 5 < self.seq_len:
            mask_1[jump2 + 5] = 1.0

        # CRITICAL ZERO-PADDING
        modality_1 = modality_1 * mask_1

        # Combine masks
        mask = torch.cat([mask_0, mask_1], dim=1)

        # Output
        x_raw = torch.cat([modality_0, modality_1], dim=1)
        return {'x_raw': x_raw, 'mask': mask, 'y_true': y_true}

if __name__ == '__main__':
    dataset = SyntheticWaddingtonDataset(size=1)
    batch = dataset[0]
    
    y_true = batch['y_true']
    x_raw = batch['x_raw']
    mask = batch['mask']
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    axes[0].plot(y_true.numpy(), color='black', linewidth=2)
    axes[0].set_title("y_true Trajectory")
    
    im1 = axes[1].imshow(x_raw.numpy().T, aspect='auto', cmap='viridis', interpolation='none')
    axes[1].set_title("x_raw Heatmap")
    
    im2 = axes[2].imshow(mask.numpy().T, aspect='auto', cmap='binary', interpolation='none')
    axes[2].set_title("mask Heatmap")
    
    plt.tight_layout()
    # Updated path to match current structure logic
    os.makedirs("output/data", exist_ok=True)
    plt.savefig("output/data/01_synthetic_data_preview.png")
    print("Saved diagnostic preview to output/data/01_synthetic_data_preview.png")
