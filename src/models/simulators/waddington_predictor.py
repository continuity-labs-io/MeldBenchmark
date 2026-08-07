import torch
import torch.nn as nn
from src.models.encoders.fusion import BiologicalCartridgeFusion
from src.models.ssm.baseline_ssm import BaselineSSM
from src.models.ssm.mask_aware_ssm import MaskAwareSSM

class WaddingtonPredictor(nn.Module):
    def __init__(self, ssm_type: str, d_cartridge: int = 30, n_modalities: int = 2, d_model: int = 64):
        super().__init__()
        self.ssm_type = ssm_type
        self.fusion = BiologicalCartridgeFusion(d_cartridge, n_modalities, d_model)
        
        if ssm_type == 'baseline':
            self.ssm = BaselineSSM(d_model)
        elif ssm_type == 'mask_aware':
            self.ssm = MaskAwareSSM(d_model)
        else:
            raise ValueError(f"Unknown ssm_type: {ssm_type}")
            
        self.readout = nn.Linear(d_model, 1)

    def forward(self, x_raw: torch.Tensor, mask: torch.Tensor):
        latent_x, latent_gate = self.fusion(x_raw, mask)
        
        if self.ssm_type == 'baseline':
            h = self.ssm(latent_x)
        elif self.ssm_type == 'mask_aware':
            h = self.ssm(latent_x, latent_gate)
            
        return self.readout(h)
