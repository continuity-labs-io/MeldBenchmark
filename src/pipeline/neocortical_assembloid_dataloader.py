import torch
from torch.utils.data import IterableDataset
import math

class NeocorticalAssembloidDataset(IterableDataset):
    """
    Simulates a multi-region thalamocortical loop assembloid generating 
    multi-modal time-series data including:
    - HD-MEA voltage traces
    - spatial optical markers
    - sparse RNA snapshots
    """
    def __init__(self, time_steps: int = 1000, latent_dim: int = 114):
        super().__init__()
        self.time_steps = time_steps
        self.latent_dim = latent_dim

    def __iter__(self):
        while True:
            # Generate homeostatic 114-D multimodal waves
            # (100D Sigma + 12D Psi + 2D Omega)
            t = torch.linspace(0, 10 * math.pi, self.time_steps).unsqueeze(1) # [time_steps, 1]
            
            # Base oscillations (homeostasis)
            freqs = torch.linspace(0.5, 5.0, self.latent_dim) # Various frequencies
            phases = torch.rand(self.latent_dim) * 2 * math.pi
            
            # Pure signal
            signal = torch.sin(t * freqs + phases)
            
            # Add biological noise
            noise = torch.randn(self.time_steps, self.latent_dim) * 0.1
            
            sequence_tensor = signal + noise # [time_steps, 114]
            
            # The dataset natively yields pure homeostasis since the metabolic crash 
            # will be manually injected during the demo.
            health_labels = torch.ones(self.time_steps)
            
            yield sequence_tensor, health_labels
