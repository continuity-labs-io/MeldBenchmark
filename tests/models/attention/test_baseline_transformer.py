import torch
import pytest
from src.models.attention.baseline_transformer import BaselineTransformer

def test_baseline_transformer_shapes():
    """Verify the transformer returns the expected shape."""
    model = BaselineTransformer(d_model=32, nhead=4, num_layers=2, max_len=100)
    
    # batch=2, seq_len=50, d_model=32
    latent_x = torch.randn(2, 50, 32)
    
    out = model(latent_x)
    assert out.shape == (2, 50, 32), f"Expected (2, 50, 32), got {out.shape}"

def test_baseline_transformer_strict_causality():
    """
    Verify the strict causal masking. A change at timestep T must have ZERO
    effect on any timestep < T.
    """
    model = BaselineTransformer(d_model=16, nhead=2, num_layers=2, max_len=100)
    model.eval()
    
    x1 = torch.randn(2, 50, 16)
    x2 = x1.clone()
    
    # Introduce a massive spike at t=25
    x2[:, 25, :] += torch.randn(2, 16) * 100.0
    
    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)
        
    # Past (t < 25): Must be absolutely identical
    diff_past = (out1[:, :25, :] - out2[:, :25, :]).abs().max().item()
    assert diff_past < 1e-5, f"Causality leak detected! Future influenced the past. Max diff: {diff_past}"
    
    # Future (t >= 25): Must diverge due to the spike
    diff_future = (out1[:, 25:, :] - out2[:, 25:, :]).abs().mean().item()
    assert diff_future > 1e-5, "Future did not diverge after the spike."
