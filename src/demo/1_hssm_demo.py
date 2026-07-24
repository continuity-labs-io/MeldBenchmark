import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import os

from src.models.state_space_engine import StateSpaceEngine
class ToyBiologicalEnvironment:
    """
    Simulates a synthetic 'Drowning Signal' multiscale biological dataset.
    Generates high-frequency GEVI data (20kHz) and lower-frequency Optical data (100Hz).
    Both modalities are corrupted by a massive 2Hz pump artifact (sine wave).
    The GEVI data contains sparse 1ms biological spikes (action potentials).

    TODO: Output Expectation to verify:
    Question: "how does it know the pump vibration isn't a biological anomaly?" 
    - The Top Panel shows the raw drowning signal (with the pump).
    - The Bottom Panel shows the Surprise metric. It will be completely flat during 
    the first 50 frames despite the massive structural wobble being fed into it, 
    proving the Fusion Core successfully zeroed it out of the predictive surprise metric.
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
    SPIKE_PROBABILITY_PER_WINDOW = 0.15
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

GEVI_COMPRESSOR_OUT_CHANNELS = 64
GEVI_COMPRESSOR_KERNEL_SIZE = 200
GEVI_COMPRESSOR_STRIDE = 200

TRAIN_ITERATIONS = 150
TRAIN_BATCH_SIZE = 16
LEARNING_RATE = 1e-3

def train_orthogonal_veto(device):
    """
    Trains the Edge Compressor (Conv1d) and Fusion Core (Mamba) jointly using 
    self-supervised predictive coding on homeostasis data.
    This forces the network to mathematically isolate spikes and orthogonalize artifacts.
    
    Args:
        device (torch.device): Device to train on.
        
    Returns:
        tuple: (gevi_compressor, mamba_engine) - The trained models.
    """
    print(f"[*] Initializing Orthogonal Veto Training on {device}...")
    
    # 1. Instantiate the "Edge Compressor"
    gevi_compressor = nn.Conv1d(
        in_channels=1, 
        out_channels=GEVI_COMPRESSOR_OUT_CHANNELS, 
        kernel_size=GEVI_COMPRESSOR_KERNEL_SIZE, 
        stride=GEVI_COMPRESSOR_STRIDE
    ).to(device)
    
    # 2. Instantiate the Fusion Core
    mamba_engine = StateSpaceEngine(
        d_model=ToyBiologicalEnvironment.OPTICS_DIM + GEVI_COMPRESSOR_OUT_CHANNELS
    ).to(device)
    
    # 3. Setup optimizer and environment
    optimizer = optim.Adam(
        list(gevi_compressor.parameters()) + list(mamba_engine.parameters()), 
        lr=LEARNING_RATE
    )
    env = ToyBiologicalEnvironment()
    
    # 4. Fast training loop
    gevi_compressor.train()
    mamba_engine.train()
    
    for iteration in range(1, TRAIN_ITERATIONS + 1):
        optimizer.zero_grad()
        
        # Generate a fresh batch
        opt_tensor, gevi_tensor = env.generate_batch(
            TRAIN_BATCH_SIZE, scenario="homeostasis", device=device
        )
        
        # Forward pass: Edge Compression
        # gevi_tensor is [Batch, 1, 20000]
        compressed_gevi = gevi_compressor(gevi_tensor) # -> [Batch, 64, 100]
        compressed_gevi = compressed_gevi.transpose(1, 2) # -> [Batch, 100, 64]
        
        # Forward pass: Fusion
        # opt_tensor is [Batch, 100, 768]
        fused_tensor = torch.cat([opt_tensor, compressed_gevi], dim=-1) # -> [Batch, 100, 832]
        
        # Forward pass: Predictive Coding
        scalar_loss, _ = mamba_engine(fused_tensor)
        
        # Backpropagation
        scalar_loss.backward()
        optimizer.step()
        
        if iteration % 30 == 0 or iteration == 1:
            print(f"    [Iteration {iteration:03d}/{TRAIN_ITERATIONS}] Loss: {scalar_loss.item():.4f}")
            
    print("[*] Training Complete.")
    return gevi_compressor, mamba_engine


def evaluate_and_plot(compressor, mamba_engine, device):
    print("[*] Generating Inference Dashboard...")
    env = ToyBiologicalEnvironment()
    
    # 2. Generate validation data
    opt_hom, gevi_hom = env.generate_batch(1, scenario="homeostasis", device=device)
    opt_cor, gevi_cor = env.generate_batch(1, scenario="corrosion", device=device)
    opt_tox, gevi_tox = env.generate_batch(1, scenario="toxic_shock", device=device)
    
    # 3. Helper to extract frame distances (Surprise)
    def get_dab(opt_tensor, gevi_tensor):
        with torch.no_grad():
            comp_gevi = compressor(gevi_tensor).transpose(1, 2)
            fused = torch.cat([opt_tensor, comp_gevi], dim=-1)
            _, frame_dists = mamba_engine(fused)
        return frame_dists.cpu().numpy()
        
    dab_hom = get_dab(opt_hom, gevi_hom)
    dab_cor = get_dab(opt_cor, gevi_cor)
    dab_tox = get_dab(opt_tox, gevi_tox)
    
    # 4. Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Top Panel: The Drowning Signal
    gevi_raw_slice = gevi_hom[0, 0, :4000].cpu().numpy()
    t_gevi_slice = torch.arange(4000).numpy() / ToyBiologicalEnvironment.GEVI_HZ
    ax1.plot(t_gevi_slice, gevi_raw_slice, color='cyan', alpha=0.8)
    ax1.set_title("The Drowning Signal (Raw 20kHz GEVI)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel("Time (s)")
    
    # Middle Panel: Orthogonal Veto (Homeostasis vs Corrosion)
    t_opt = torch.arange(len(dab_hom)).numpy()
    ax2.plot(t_opt, dab_hom, label="Homeostasis", color='green', linewidth=2)
    ax2.plot(t_opt, dab_cor, label="Corrosion (Hardware Failure)", color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=ToyBiologicalEnvironment.EVENT_BOUNDARY_OPTICS, color='gray', linestyle='--', label="Event Boundary (T=50)")
    ax2.set_title("Orthogonal Veto (Surprise)")
    ax2.set_ylabel("Surprise (Cosine Distance)")
    ax2.set_xlabel("Time (Optical Frames)")
    ax2.legend()
    
    # Bottom Panel: True Crash (Toxic Shock)
    ax3.plot(t_opt, dab_tox, label="Toxic Shock (Biological Crash)", color='purple', linewidth=2)
    ax3.axvline(x=ToyBiologicalEnvironment.EVENT_BOUNDARY_OPTICS, color='gray', linestyle='--', label="Event Boundary (T=50)")
    ax3.set_title("True Crash Detection (Surprise)")
    ax3.set_ylabel("Surprise (Cosine Distance)")
    ax3.set_xlabel("Time (Optical Frames)")
    ax3.legend()
    
    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/1_hssm_veto_proof.png")
    print("[*] Dashboard saved to output/1_hssm_veto_proof.png")

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
    
    print("\n[*] Testing Training Loop...")
    gevi_comp, mamba = train_orthogonal_veto(device)

    print("\n[*] Generating Dashboard...")
    evaluate_and_plot(gevi_comp, mamba, device)

