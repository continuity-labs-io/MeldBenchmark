import torch
import torch.nn as nn
import torch.nn.functional as F

class MeldLoss(nn.Module):
    """
    Composite loss function for the MELD Large Biological Model (LBM) state-space training loop.
    Incorporates Next-Frame Forecasting, Lipschitz continuous penalty, and Time-Reversal Error.
    """
    def __init__(self, alpha=1.0, beta=0.1, gamma=1.0, L=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.L = L

    def forward(self, state_t, target_t_plus_1, pred_t_plus_1, reconstructed_t, delta_x):
        """
        Calculates the composite loss.

        Args:
            state_t: Tensor of shape (batch_size, ...) representing the actual state at time t.
            target_t_plus_1: Tensor of shape (batch_size, ...) representing the ground-truth state at time t+1.
            pred_t_plus_1: Tensor of shape (batch_size, ...) representing the predicted state at time t+1.
            reconstructed_t: Tensor of shape (batch_size, ...) representing the reconstructed state at time t.
            delta_x: Tensor of shape (batch_size, 1) representing the magnitude of the perturbation/time step.

        Returns:
            Tuple containing:
            - L_total: A scalar tensor representing the total weighted loss.
            - metrics: A dictionary of individual detached loss components for telemetry logging.
        """
        # 1. Next-Frame Forecasting (L_forecast)
        l_forecast = F.mse_loss(pred_t_plus_1, target_t_plus_1)

        # 2. Lipschitz Penalty (L_lipschitz)
        # Calculate the predicted state change: Δy = pred_t_plus_1 - state_t
        delta_y = pred_t_plus_1 - state_t
        
        # Calculate the L2 norm of the predicted state change per sample across all non-batch dimensions
        batch_size = delta_y.size(0)
        delta_y_flat = delta_y.view(batch_size, -1)
        norm_delta_y = torch.norm(delta_y_flat, p=2, dim=1, keepdim=True) # shape (batch_size, 1)

        # Penalize this norm if it exceeds L * delta_x: max(0, ||Δy|| - L * Δx)
        lipschitz_violations = F.relu(norm_delta_y - self.L * delta_x)
        
        # Mean across the batch
        l_lipschitz = lipschitz_violations.mean()

        # 3. Time-Reversal Error (L_reverse)
        l_reverse = F.mse_loss(reconstructed_t, state_t)

        # Total Loss
        l_total = self.alpha * l_forecast + self.beta * l_lipschitz + self.gamma * l_reverse

        # Telemetry Dictionary
        metrics = {
            "forecast_loss": l_forecast.detach().item(),
            "lipschitz_penalty": l_lipschitz.detach().item(),
            "reverse_loss": l_reverse.detach().item(),
        }

        return l_total, metrics
