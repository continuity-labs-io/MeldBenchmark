import pytest
import torch

try:
    from src.models.encoders.topo_encoder import TopoEncoder

    HAS_MAMBA = True
except Exception:
    HAS_MAMBA = False


@pytest.mark.skipif(not HAS_MAMBA, reason="Mamba not available or failed to load")
def test_topo_encoder():
    try:
        model = TopoEncoder(d_model=64, d_state=16, d_conv=4, expand=2)
    except Exception as e:
        pytest.skip(f"Mamba instantiation failed: {e}")

    batch = 2
    time = 5
    # Input shape: [Batch, Time, 2, 64, 64]
    x = torch.randn(batch, time, 2, 64, 64)

    # Test without hidden states
    out = model(x, return_hidden=False)
    assert out.shape == (batch, 64)

    # Test with hidden states
    out, hidden = model(x, return_hidden=True)
    assert out.shape == (batch, 64)
    assert hidden.shape == (batch, time, 64)
