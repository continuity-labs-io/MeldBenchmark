import torch
import torch.nn as nn


class BiologicalCartridgeFusion(nn.Module):
    """
    Fusion Encoder for Biological Cartridge.

    A 'cartridge' acts as an abstraction of a biological sample providing
    telemetry or similar multimodal data. This encoder projects hardware data
    to dense space and routes the boolean mask to the latent subspace.
    """

    def __init__(self, d_cartridge: int, n_modalities: int, d_model: int):
        super().__init__()
        self.W_cart = nn.Linear(d_cartridge, d_model, bias=False)
        self.W_gate = nn.Linear(n_modalities, d_model, bias=True)

    def forward(self, x_raw: torch.Tensor, mask: torch.Tensor):
        latent_x = self.W_cart(x_raw)
        latent_gate = torch.sigmoid(self.W_gate(mask))
        return latent_x, latent_gate
