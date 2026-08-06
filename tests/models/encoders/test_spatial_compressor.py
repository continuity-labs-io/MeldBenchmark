import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

# We patch timm before importing SpatialCompressor if possible, 
# or patch it around the instantiation.
from src.models.encoders.spatial_compressor import SpatialCompressor

def test_spatial_compressor():
    # Create a dummy ViT model that outputs [B*T, 768]
    dummy_vit = MagicMock()
    # The real ViT returns a tensor of shape [B*T, 768]
    # We use side_effect to dynamically return the correct shape based on input shape
    def dummy_forward(x_flat):
        b_t = x_flat.shape[0]
        return torch.randn(b_t, 768)
        
    dummy_vit.side_effect = dummy_forward
    # Need to mock parameters() for the freezing loop
    dummy_param = nn.Parameter(torch.empty(0))
    dummy_vit.parameters.return_value = [dummy_param]
    
    with patch('timm.create_model', return_value=dummy_vit):
        model = SpatialCompressor()
        
        batch = 2
        time = 5
        # Input shape: [Batch, Time, Channels, Depth, Height, Width]
        # Our model expects depth=dim3, channels=2
        x = torch.randn(batch, time, 2, 32, 64, 64)
        
        out = model(x)
        
        assert out.shape == (batch, time, 768)
        assert not dummy_param.requires_grad
