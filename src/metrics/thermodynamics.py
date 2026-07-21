import torch
import torch.nn.functional as F

class ThermodynamicMetrics:
    def __init__(self, alpha=1000.0, beta=1.0):
        """
        Calculates kinetic biomarkers from the continuous biological latent space.
        """
        self.alpha = alpha
        self.beta = beta

    def calculate_cvi(self, z_sequence, window_size=3):
        """
        Critical Variance Index (CVI)
        Tracks the physical 'wobble' (Variance) and sluggishness (AR1) of the cell.
        z_sequence shape: [Time, Embed_Dim]
        """
        time_steps = z_sequence.shape[0]
        cvi_scores = []

        # Graceful fallback if sequence is too short
        if time_steps < window_size:
            return [0.0] * max(1, time_steps)

        for t in range(window_size, time_steps + 1):
            z_win = z_sequence[t-window_size:t, :]
            
            # Variance (The Wobble)
            var_t = torch.var(z_win, dim=0).mean().item()
            
            # Lag-1 Autocorrelation (Critical Slowing Down)
            ar1_t = F.cosine_similarity(z_win[:-1, :], z_win[1:, :], dim=1).mean().item()
            
            cvi = (self.alpha * var_t) + (self.beta * ar1_t)
            cvi_scores.append(cvi)

        # Pad initial frames to maintain temporal sequence length
        return [cvi_scores[0]] * (window_size - 1) + cvi_scores

    def calculate_dab(self, z_sequence):
        """
        Distance-to-Absorbing-Boundary (DAB)
        Empirically approximates the Jacobian of the latent transition (J = dz_{t+1} / dz_t).
        Extracts the dominant eigenvalue using Power Iteration.
        As the eigenvalue -> 1.0, the attractor basin flattens (Saddle-Node Bifurcation).
        """
        time_steps = z_sequence.shape[0]
        dab_scores = [1.0] # Starts at healthy 1.0 (max distance from cliff)
        
        for t in range(1, time_steps):
            v_in = z_sequence[t-1, :].unsqueeze(0)   # [1, Embed_Dim]
            v_out = z_sequence[t, :].unsqueeze(0)    # [1, Embed_Dim]
            
            # Empirical low-rank approximation of the Jacobian
            v_in_norm = F.normalize(v_in, dim=1)
            v_out_norm = F.normalize(v_out, dim=1)
            J_approx = torch.matmul(v_out_norm.T, v_in_norm) # [Embed_Dim, Embed_Dim]
            
            # Calculate the dominant eigenvalue using fast power iteration
            num_iters = 5
            b_k = torch.randn(J_approx.shape[1], 1).to(z_sequence.device)
            for _ in range(num_iters):
                b_k1 = torch.matmul(J_approx, b_k)
                b_k = b_k1 / (torch.norm(b_k1) + 1e-8)
                
            # Rayleigh quotient gives the dominant eigenvalue
            eigenvalue = torch.matmul(b_k.T, torch.matmul(J_approx, b_k)) / (torch.matmul(b_k.T, b_k) + 1e-8)
            
            # The mathematical distance to the crash
            dab = 1.0 - abs(eigenvalue.item())
            dab_scores.append(max(0.0, dab))
            
        return dab_scores
        
    def calculate_hysteresis(self, z_stress, z_rescue):
        """
        Transcriptomic / Morphological Hysteresis
        Calculates the topological area trapped between the stress path and rescue path.
        Hysteresis = \oint \vec{x}(p) dp
        """
        # Ensure temporal lengths match for integration
        min_steps = min(z_stress.shape[0], z_rescue.shape[0])
        path_down = z_stress[:min_steps, :]
        path_up = z_rescue[:min_steps, :]
        
        # Calculate the Euclidean distance between the paths at every time step
        path_divergence = torch.norm(path_down - path_up, dim=1)
        
        # Integrate the area under the divergence curve using the Trapezoidal Rule
        hysteresis_area = torch.trapz(path_divergence).item()
        
        return hysteresis_area
