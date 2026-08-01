import pytest
import torch
import os
import sys

# Mamba requires specific architecture compilation which might fail on some minimal CI environments
# We will gracefully skip if Mamba instantiation fails
try:
    from src.models.state_space_engine import StateSpaceEngine
    HAS_MAMBA = True
except Exception:
    HAS_MAMBA = False

@pytest.mark.skipif(not HAS_MAMBA, reason="Mamba not available or failed to load")
def test_state_space_engine():
    # Use lightweight settings to avoid memory issues
    try:
        model = StateSpaceEngine(d_model=64, d_state=16, d_conv=4, expand=2)
    except Exception as e:
        pytest.skip(f"Mamba instantiation failed (likely architecture mismatch): {e}")

    batch_size = 2
    time_steps = 10
    d_model = 64
    x = torch.randn(batch_size, time_steps, d_model)
    
    scalar_loss, frame_distances = model(x)
    
    assert isinstance(scalar_loss, torch.Tensor)
    assert scalar_loss.ndim == 0
    # frame_distances is [Time - 1]
    assert frame_distances.shape == (time_steps - 1,)
