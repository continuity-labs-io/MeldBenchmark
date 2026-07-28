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
        The code snippet implements Dynamic Mode Decomposition using a truncated Singular Value Decomposition. 
        By decomposing the sliding window of latent states X, the algorithm approximates the local linear 
        operator A_tilde that steps the system forward in time to state Y. The eigenvalues of this operator 
        directly quantify the thermodynamic stability of the biological system. A maximum eigenvalue near 1.0 
        indicates stable homeostasis, while a diverging eigenvalue maps to the system crossing the 
        absorbing boundary into a structural crash.

        The decision to utilize Dynamic Mode Decomposition over pseudo-arc length continuation stems from 
        the specific architectural constraints of the platform.
        - Compute Latency: Pseudo-arc length continuation is an iterative root-finding algorithm. 
        It is computationally expensive and risks introducing variable execution times.
        Truncated Singular Value Decomposition over a small sliding window is deterministic and executes 
        with high efficiency on edge GPUs, ensuring the metric keeps pace with the biological timescales 
        of milliseconds to minutes.
        - Model Independence: Pseudo-arc length continuation requires an explicit, differentiable 
        non-linear vector field to compute the Jacobian. Because the tissue trajectory is modeled 
        inside the continuous latent space of the state-space engine, defining the exact non-linear
         continuous field is complex. 
         Dynamic Mode Decomposition is entirely data-driven, extracting the kinetic modes directly from
          the streaming embeddings without requiring the underlying equations.
        - Architectural Simplicity: The current approach provides a fast, elegant solution that 
        satisfies the requirement for a real-time predictive metric. It isolates the critical variance 
        and successfully detects the Waddington bifurcation point while keeping the codebase lean.
        """
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

    def calculate_lle(self, z_sequence, window_size=4, dt=1.0):
        """
        Computes the Local Lyapunov Exponent (LLE) over a sliding window
        to measure the stability of the biological attractor basin.
        """
        import math

        time_steps = z_sequence.shape[0]
        lle_scores = [0.0] * window_size
        if time_steps <= window_size:
            return [0.0] * time_steps

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

            # Calculate LLE
            lle = math.log(max_eig + 1e-7) / dt
            lle_scores.append(lle)
            
        return lle_scores

    def calculate_cka(self, z_seq1, z_seq2):
        """
        Calculates the Linear Centered Kernel Alignment (CKA) to prove that the geometric shape 
        of the biological manifold is preserved across multi-day recordings, even in the presence 
        of representational drift.
        """
        min_steps = min(z_seq1.shape[0], z_seq2.shape[0])
        
        # Trim to minimum length
        X = z_seq1[:min_steps, :]
        Y = z_seq2[:min_steps, :]
        
        n = min_steps
        device = X.device
        
        # Compute the linear Gram matrices
        K = X @ X.T
        L = Y @ Y.T
        
        # Center the Gram matrices
        H = torch.eye(n, device=device) - (torch.ones(n, n, device=device) / n)
        K_c = H @ K @ H
        L_c = H @ L @ H
        
        # Compute the Hilbert-Schmidt Independence Criterion (HSIC)
        def hsic(A, B):
            return torch.trace(A @ B)
            
        hsic_kl = hsic(K_c, L_c)
        hsic_kk = hsic(K_c, K_c)
        hsic_ll = hsic(L_c, L_c)
        
        # Return the normalized CKA score
        cka = hsic_kl / torch.sqrt(hsic_kk * hsic_ll)
        return cka.item()
