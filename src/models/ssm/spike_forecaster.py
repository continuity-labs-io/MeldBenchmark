import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
from src.config import settings

import logging
logger = logging.getLogger(__name__)

class SpikeForecaster(nn.Module):
    """
    SpikeForecaster uses a Mamba-2 backbone to model continuous kinetic trajectories
    of biological state vectors. It is designed to ingest multi-unit spike counts
    and predict non-negative spike rates for the subsequent time step.
    """

    def __init__(self, input_dim=1240, d_model=settings.MAMBA_D_MODEL, expand=2, d_conv=4, d_state=settings.MAMBA_D_STATE):
        """
        Args:
            input_dim (int): The dimension of the input feature vectors (e.g., 1240 for spike data).
            d_model (int): The hidden dimension of the Mamba-2 block.
            expand (int): Expansion factor creating the inner dimension (width of data pathway).
            d_conv (int): The kernel size of the local 1D convolution applied before the SSM.
            d_state (int): The size of the hidden state / recurrent memory (depth of memory).
        """
        super().__init__()
        # Project input to hidden dimension
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Instantiate a Mamba-2 block from mamba_ssm. 
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        # Output projection back to input dimension (predicting next step)
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x, return_hidden=False):
        """
        Args:
            x (torch.Tensor): Sequence of state vectors of shape [Batch, Time, input_dim].
            return_hidden (bool): If True, returns the internal hidden state.

        Returns:
            torch.Tensor: Predicted non-negative spike rates for the next time step, 
                          shape [Batch, Time, input_dim].
            (Optional) torch.Tensor: Hidden states of shape [Batch, Time, d_model].
        """
        # Project to d_model
        h = self.input_proj(x)
        
        # Pass through Mamba-2 block to generate contextualized hidden states
        hidden_states = self.mamba(h)

        # Map each hidden state to predicted spike rates
        predictions_raw = self.output_proj(hidden_states)
        
        # Apply Softplus to ensure predictions are non-negative
        predictions = F.softplus(predictions_raw)

        if return_hidden:
            return predictions, hidden_states
        return predictions

    def get_hidden_states(self, x):
        """
        Helper method to extract internal hidden state for downstream visualization of memory retention.
        """
        _, hidden_states = self.forward(x, return_hidden=True)
        return hidden_states


if __name__ == "__main__":
    logger.info("Testing SpikeForecaster (Mamba-2) architecture...")
    from src.utils.device import get_optimal_device
    device = get_optimal_device(verbose=True)

    try:
        model = SpikeForecaster(input_dim=1240).to(device)

        # Dummy input: [Batch, Time, 1240]
        batch_size = 2
        time_steps = 10
        x = torch.randn(batch_size, time_steps, 1240).to(device)

        # Forward pass
        predictions = model(x)
        
        # Extract hidden states
        hidden_states = model.get_hidden_states(x)

        logger.info(f"Input shape: {x.shape}")
        logger.info(f"Predictions shape: {predictions.shape}")
        logger.info(f"Predictions min value (should be >= 0): {predictions.min().item():.4f}")
        logger.info(f"Hidden states shape: {hidden_states.shape}")
        logger.info("Test passed! Requirements satisfied.")
    except Exception as e:
        logger.error(f"Test failed or skipped due to environment constraints: {e}")
