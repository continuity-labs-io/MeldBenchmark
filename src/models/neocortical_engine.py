import torch
import torch.nn as nn
from mamba_ssm import Mamba2

class NeocorticalEngine(nn.Module):
    """
    Unifies the Mamba-2 sequential forecaster with a thermodynamic state-space bridge
    and an interpretability attribution hook.
    """
    def __init__(self, input_dim=114, d_model=768, d_state=64):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Project multimodal state vector to d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Stack of two Mamba2 blocks
        self.mamba_block1 = Mamba2(d_model=d_model, d_state=d_state)
        self.mamba_block2 = Mamba2(d_model=d_model, d_state=d_state)
        
        # Forward predictive coding head (predicts t+1)
        self.forward_head = nn.Linear(d_model, input_dim)
        
        # Reverse reconstruction head (reconstructs t)
        self.reverse_head = nn.Linear(d_model, input_dim)
        
    def forward(self, x, return_hidden=False):
        """
        x: [batch, time, input_dim]
        Returns:
            predicted_state: [batch, time, input_dim]
            reconstructed_state: [batch, time, input_dim]
            (optional) hidden: [batch, time, d_model]
        """
        # Project
        h = self.input_proj(x) # [batch, time, d_model]
        
        # Sequence modeling
        h = self.mamba_block1(h)
        h = self.mamba_block2(h)
        
        # Forward prediction and reverse reconstruction
        predicted_state = self.forward_head(h) # predicting t+1
        reconstructed_state = self.reverse_head(h) # predicting t
        
        if return_hidden:
            return predicted_state, reconstructed_state, h
        return predicted_state, reconstructed_state

    def compute_attribution(self, x, target_time_step):
        """
        Executes First-Order Taylor Decomposition (Input * Gradient) on the predicted 
        state vector at `target_time_step`.
        
        x: [batch, time, input_dim]
        target_time_step: int
        
        Returns:
            attribution_matrix: [batch, time, input_dim]
        """
        x_req = x.clone().detach().requires_grad_(True)
        
        predicted_state, _ = self.forward(x_req)
        
        # Target state at target_time_step
        # Sum over feature dimension to get a scalar proxy for the whole state at that time step
        target_state_sum = predicted_state[:, target_time_step, :].sum()
        
        # Compute gradient of target state w.r.t input x
        gradients = torch.autograd.grad(target_state_sum, x_req, retain_graph=True)[0]
        
        # First-Order Taylor Decomposition: Input * Gradient
        attribution_matrix = x_req.detach() * gradients
        
        return attribution_matrix
