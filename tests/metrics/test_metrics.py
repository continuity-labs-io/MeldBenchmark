import pytest
import torch
from src.metrics.metrics import ThermodynamicMetrics

@pytest.fixture
def metrics_engine():
    return ThermodynamicMetrics()

def test_calculate_csd(metrics_engine):
    # Shape: [Time, Embed_Dim]
    z_seq = torch.randn(20, 16)
    csd_scores = metrics_engine.calculate_csd(z_seq, window_size=5)
    
    assert len(csd_scores) == 20
    assert all(isinstance(score, float) for score in csd_scores)

def test_calculate_ksm(metrics_engine):
    z_seq = torch.randn(20, 16)
    ksm_scores = metrics_engine.calculate_ksm(z_seq, window_size=5)
    
    assert len(ksm_scores) == 20
    assert all(0.0 <= score <= 1.0 for score in ksm_scores)

def test_calculate_hysteresis(metrics_engine):
    z_baseline = torch.randn(20, 16)
    z_perturbed = torch.randn(20, 16)
    
    area, divergence = metrics_engine.calculate_hysteresis(z_baseline, z_perturbed)
    
    assert isinstance(area, float)
    assert len(divergence) == 20
    assert area >= 0.0

def test_calculate_lle(metrics_engine):
    z_seq = torch.randn(20, 16)
    lle_scores = metrics_engine.calculate_lle(z_seq, window_size=5)
    
    assert len(lle_scores) == 20
    assert all(isinstance(score, float) for score in lle_scores)

def test_calculate_cka(metrics_engine):
    z_seq1 = torch.randn(20, 16)
    z_seq2 = torch.randn(20, 16)
    
    cka_score = metrics_engine.calculate_cka(z_seq1, z_seq2)
    assert isinstance(cka_score, float)
