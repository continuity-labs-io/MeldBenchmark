import torch
import torch.nn as nn

class BaselineSSM(nn.Module):
    """
    Standard continuous-time SSM that ingests data blindly. Acts as a control group.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model) * 0.5 + 0.1))
        self.B_proj = nn.Linear(d_model, d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model)

    def forward(self, latent_x: torch.Tensor):
        """
        Args:
            latent_x: Tensor of shape [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = latent_x.size()
        h_prev = torch.zeros(batch, d_model, device=latent_x.device)
        
        A = -torch.exp(self.A_log)
        
        hidden_states = []
        for t in range(seq_len):
            x_t = latent_x[:, t, :]
            dt = torch.nn.functional.softplus(self.dt_proj(x_t))
            B = self.B_proj(x_t)
            
            A_bar = torch.exp(A * dt)
            B_bar = (A_bar - 1.0) / (A - 1e-8) * B
            
            h_prev = A_bar * h_prev + B_bar
            hidden_states.append(h_prev)
            
        return torch.stack(hidden_states, dim=1)
