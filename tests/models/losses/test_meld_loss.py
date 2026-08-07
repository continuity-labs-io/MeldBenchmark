import pytest
import torch
from src.models.losses.meld_loss import MeldLoss, TopoContrastiveLoss


def test_meld_loss():
    loss_fn = MeldLoss()
    batch_size = 2
    embed_dim = 16

    state_t = torch.randn(batch_size, embed_dim)
    target_t_plus_1 = torch.randn(batch_size, embed_dim)
    pred_t_plus_1 = torch.randn(batch_size, embed_dim)
    reconstructed_t = torch.randn(batch_size, embed_dim)
    delta_x = torch.ones(batch_size, 1) * 0.1

    l_total, metrics = loss_fn(state_t, target_t_plus_1, pred_t_plus_1, reconstructed_t, delta_x)

    assert isinstance(l_total, torch.Tensor)
    assert l_total.ndim == 0  # scalar
    assert "forecast_loss" in metrics
    assert "lipschitz_penalty" in metrics
    assert "reverse_loss" in metrics


def test_topo_contrastive_loss():
    loss_fn = TopoContrastiveLoss()
    batch_size = 2
    embed_dim = 16

    lfp_latents = torch.randn(batch_size, embed_dim)
    vision_latents = torch.randn(batch_size, embed_dim)

    loss, metrics = loss_fn(lfp_latents, vision_latents)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar
    assert "contrastive_loss" in metrics
