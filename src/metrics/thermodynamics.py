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
        """
        Distance-to-Absorbing-Boundary (DAB) via Dynamic Mode Decomposition (DMD)
        Approximates the local Jacobian A where Z_{future} = Z_{past} A
        Extracts the dominant eigenvalue. As eigenvalue -> 1.0, DAB -> 0.0 (Crash).
        """
        time_steps = z_sequence.shape[0]
        dab_scores = [1.0] * window_size  # Start healthy

        if time_steps <= window_size:
            return [1.0] * time_steps

        for t in range(window_size, time_steps):
            X = z_sequence[t - window_size : t]  # Past states
            Y = z_sequence[t - window_size + 1 : t + 1]  # Future states

            # We want A such that X @ A = Y.
            # Using SVD on X to find the pseudoinverse efficiently: X = U @ S @ Vh
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)

            # The non-zero eigenvalues of the full [Embed_Dim, Embed_Dim] operator A
            # are exactly the eigenvalues of the smaller [window_size, window_size] projected
            # matrix A_tilde.
            S_inv = torch.diag(S / (S**2 + 1e-4))
            A_tilde = S_inv @ U.T @ Y @ Vh.T

            # Extract eigenvalues
            eigenvalues = torch.linalg.eigvals(A_tilde)
            max_eig = torch.max(torch.abs(eigenvalues)).item()

            # DAB is bounded [0, 1]. A stable system has max_eig near 1.0.
            dab = 1.0 / (1.0 + abs(max_eig - 1.0))
            dab_scores.append(dab)

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
