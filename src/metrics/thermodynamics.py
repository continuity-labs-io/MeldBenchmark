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
            z_win = z_sequence[t - window_size : t, :]

            # Variance (The Wobble)
            var_t = torch.var(z_win, dim=0).mean().item()

            # Lag-1 Autocorrelation (Critical Slowing Down)
            ar1_t = F.cosine_similarity(z_win[:-1, :], z_win[1:, :], dim=1).mean().item()

            cvi = (self.alpha * var_t) + (self.beta * ar1_t)
            cvi_scores.append(cvi)

        # Pad initial frames to maintain temporal sequence length
        return [cvi_scores[0]] * (window_size - 1) + cvi_scores

    def calculate_dab(self, z_sequence, window_size=4):
        import math

        time_steps = z_sequence.shape[0]
        dab_scores = [1.0] * window_size
        if time_steps <= window_size:
            return [1.0] * time_steps

        for t in range(window_size, time_steps):
            X = z_sequence[t - window_size : t]
            Y = z_sequence[t - window_size + 1 : t + 1]

            U, S, Vh = torch.linalg.svd(X, full_matrices=False)

            # TRUNCATED SVD: Keep only significant singular values
            rank = max(1, (S > 1e-5 * S[0]).sum().item())
            U_k = U[:, :rank]
            S_inv_k = torch.diag(1.0 / S[:rank])
            Vh_k = Vh[:rank, :]

            A_tilde = S_inv_k @ U_k.T @ Y @ Vh_k.T
            eigenvalues = torch.linalg.eigvals(A_tilde)
            max_eig = torch.max(torch.abs(eigenvalues)).item()

            # Bound DAB smoothly [0, 1] using an exponential envelope
            dab = math.exp(-0.5 * abs(max_eig - 1.0))
            dab_scores.append(max(0.0, dab))
        return dab_scores

    def calculate_hysteresis(self, z_baseline, z_perturbed):
        """
        Morphological Hysteresis
        Calculates the topological area between the stress path and rescue path.
        """
        min_steps = min(z_baseline.shape[0], z_perturbed.shape[0])
        if min_steps < 2:
            return 0.0, []

        path_down = z_baseline[:min_steps, :]
        path_up = z_perturbed[:min_steps, :]

        # Euclidean distance between the paths at every time step
        path_divergence = torch.norm(path_down - path_up, dim=1)

        # Integrate the area under the divergence curve using the Trapezoidal Rule
        hysteresis_area = torch.trapz(path_divergence).item()

        return hysteresis_area, path_divergence.tolist()
