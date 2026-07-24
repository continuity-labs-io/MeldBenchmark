import torch
import math

class ToyBiologicalEnvironment:
    """
    Simulates a synthetic 'Drowning Signal' multiscale biological dataset.
    Generates high-frequency GEVI data (20kHz) and lower-frequency Optical data (100Hz).
    Both modalities are corrupted by a massive 2Hz pump artifact (sine wave).
    The GEVI data contains sparse 1ms biological spikes (action potentials).
    """

    # --- Constants for Data Generation ---
    OPTICS_HZ = 100
    GEVI_HZ = 20000
    DURATION_SECONDS = 1.0

    OPTICS_FRAMES = int(OPTICS_HZ * DURATION_SECONDS)  # 100 frames
    GEVI_FRAMES = int(GEVI_HZ * DURATION_SECONDS)      # 20000 frames
    OPTICS_DIM = 768
    GEVI_DIM = 1

    PUMP_ARTIFACT_HZ = 2.0
    OPTICS_PUMP_AMPLITUDE = 2.0
    GEVI_PUMP_AMPLITUDE = 50.0

    SPIKE_AMPLITUDE = 100.0
    SPIKE_PROBABILITY_PER_WINDOW = 0.01
    SPIKE_WINDOW_STEPS = int(GEVI_HZ / OPTICS_HZ) # 200 steps per optic frame
    SPIKE_WIDTH_STEPS = int(GEVI_HZ * 0.001)      # 1ms spike = 20 steps

    EVENT_BOUNDARY_OPTICS = 50
    EVENT_BOUNDARY_GEVI = EVENT_BOUNDARY_OPTICS * SPIKE_WINDOW_STEPS # 10000
    TOXIC_SHOCK_NOISE_STD = 5.0
    CORROSION_DRIFT_STD = 2.0

    def __init__(self):
        pass

    def generate_batch(self, batch_size, scenario="homeostasis", device="cpu"):
        """
        Generates a synthetic batch of multiscale data.

        Args:
            batch_size (int): Number of independent sequences to generate.
            scenario (str): One of 'homeostasis', 'corrosion', or 'toxic_shock'.
            device (str or torch.device): Target device ('cpu', 'cuda', 'mps').

        Returns:
            tuple:
                - optical_tensor (torch.Tensor): Shape [batch_size, 100, 768], float32
                - gevi_tensor (torch.Tensor): Shape [batch_size, 1, 20000], float32
        """
        # 1. Base Time Tensors
        t_optics = torch.arange(self.OPTICS_FRAMES, device=device, dtype=torch.float32) / self.OPTICS_HZ
        t_gevi = torch.arange(self.GEVI_FRAMES, device=device, dtype=torch.float32) / self.GEVI_HZ

        # 2. The Pump Artifact (2Hz sine wave)
        # Optical Pump Artifact: [100] -> [1, 100, 1]
        pump_optics = self.OPTICS_PUMP_AMPLITUDE * torch.sin(2 * math.pi * self.PUMP_ARTIFACT_HZ * t_optics)
        pump_optics = pump_optics.view(1, self.OPTICS_FRAMES, 1)

        # GEVI Pump Artifact: [20000] -> [1, 1, 20000]
        pump_gevi = self.GEVI_PUMP_AMPLITUDE * torch.sin(2 * math.pi * self.PUMP_ARTIFACT_HZ * t_gevi)
        pump_gevi = pump_gevi.view(1, 1, self.GEVI_FRAMES)

        # Initialize base tensors with the pump artifacts
        optical_tensor = pump_optics.expand(batch_size, self.OPTICS_FRAMES, self.OPTICS_DIM).clone()
        gevi_tensor = pump_gevi.expand(batch_size, self.GEVI_DIM, self.GEVI_FRAMES).clone()

        # 3. The Biology (Sparse 1ms spikes)
        # We iterate over the 200-step windows and randomly inject spikes.
        num_windows = self.GEVI_FRAMES // self.SPIKE_WINDOW_STEPS
        
        for b in range(batch_size):
            for w in range(num_windows):
                # Under toxic shock, biology completely stops after T=50
                if scenario == "toxic_shock" and w >= self.EVENT_BOUNDARY_OPTICS:
                    continue
                
                # Check if a spike occurs in this window
                if torch.rand(1).item() < self.SPIKE_PROBABILITY_PER_WINDOW:
                    # Choose a random start within the window
                    start_idx = w * self.SPIKE_WINDOW_STEPS + torch.randint(
                        0, self.SPIKE_WINDOW_STEPS - self.SPIKE_WIDTH_STEPS, (1,)
                    ).item()
                    end_idx = start_idx + self.SPIKE_WIDTH_STEPS
                    gevi_tensor[b, 0, start_idx:end_idx] += self.SPIKE_AMPLITUDE

        # 4 & 5 & 6. Scenario Modifications
        if scenario == "corrosion":
            # Hardware Failure: massive random-walk baseline drift on GEVI after T=50 (10000 steps)
            # Optical remains perfectly normal.
            drift_steps = self.GEVI_FRAMES - self.EVENT_BOUNDARY_GEVI
            
            # Generate random step sizes and cumulatively sum them to create a random walk
            random_steps = torch.randn((batch_size, 1, drift_steps), device=device) * self.CORROSION_DRIFT_STD
            random_walk = torch.cumsum(random_steps, dim=-1)
            
            gevi_tensor[:, :, self.EVENT_BOUNDARY_GEVI:] += random_walk

        elif scenario == "toxic_shock":
            # Biological Crash: GEVI spikes stopped (handled in loop above).
            # Optical tensor experiences a variance explosion.
            noise_steps = self.OPTICS_FRAMES - self.EVENT_BOUNDARY_OPTICS
            variance_explosion = torch.randn((batch_size, noise_steps, self.OPTICS_DIM), device=device) * self.TOXIC_SHOCK_NOISE_STD
            
            optical_tensor[:, self.EVENT_BOUNDARY_OPTICS:, :] += variance_explosion

        return optical_tensor, gevi_tensor

if __name__ == "__main__":
    # Quick sanity check
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    env = ToyBiologicalEnvironment()
    
    opt, gev = env.generate_batch(4, scenario="homeostasis", device=device)
    print(f"Homeostasis -> Optics: {opt.shape}, GEVI: {gev.shape}")
    
    opt, gev = env.generate_batch(4, scenario="corrosion", device=device)
    print(f"Corrosion   -> Optics: {opt.shape}, GEVI: {gev.shape}")
    
    opt, gev = env.generate_batch(4, scenario="toxic_shock", device=device)
    print(f"Toxic Shock -> Optics: {opt.shape}, GEVI: {gev.shape}")
