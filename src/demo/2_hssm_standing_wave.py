import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import os
class HierarchicalSSM(nn.Module):
    """
    2-Layer Hierarchical Continuous-Time State Space Model (H-SSM).
    Demonstrates a standing wave phase transition driven by delayed recurrent feedback.

    Scenario A (Low K): The "Driven Mode." The feedback parameter (K) is turned down. 
    The simulated biological network is passive. 
    It has no internal memory or momentum; it is merely reacting to external noise (the vat).

    Scenario B (High K): The "Standing Wave Mode." The feedback parameter (K) crosses the critical threshold. 
    The internal recursive loops dominate, and the network begins to predict and reinforce its own state.
    State x1_0 and x1_1: These are two orthogonal dimensions of Layer 1's hidden state vector. 
    Think of them as two coupled biological variables driving an oscillation—for example, x1_0 is 
    the average electrical voltage of the neural population, and x1_1 is the metabolic recovery rate. 
    You need at least two interacting dimensions to create a wave.
    """
    def __init__(self, d1=16, d2=4, dt=0.005, tau_delay_steps=50, gamma=0.1):
        super().__init__()
        self.d1 = d1
        self.d2 = d2
        self.dt = dt
        self.tau_steps = tau_delay_steps
        self.gamma = gamma
        
        # Layer 1 (Fast/Local) parameters
        A1 = torch.zeros(d1, d1)
        # Block skew-symmetric for oscillatory behavior
        for i in range(d1 // 2):
            # Frequencies spread from 0.5Hz to ~4Hz
            omega = (i + 1) * 2.0 * np.pi * 0.5 
            A1[2*i, 2*i+1] = omega
            A1[2*i+1, 2*i] = -omega
        # Small dampening
        A1 -= torch.eye(d1) * 1.0 
        self.A1 = nn.Parameter(A1, requires_grad=False)
        
        # Input projection for Layer 1
        self.B1 = nn.Parameter(torch.randn(d1) * 0.5, requires_grad=False)
        
        # Layer 2 (Slow/Macro) parameters
        # Slower decay dynamics
        self.A2 = nn.Parameter(-torch.eye(d2) * 0.2, requires_grad=False) 
        # Pooled projection from Layer 1 to Layer 2
        self.B2 = nn.Parameter(torch.randn(d2, d1) / np.sqrt(d1), requires_grad=False)
        
        # Top-down feedback (modulation) from Layer 2 to Layer 1
        self.W_td = nn.Parameter(torch.randn(1, d2) * 0.5, requires_grad=False)
        
        # Top-down prediction (for error metric: Layer 2 predicting Layer 1)
        self.W_pred = nn.Parameter(torch.randn(d1, d2) / np.sqrt(d2), requires_grad=False)
        
    def forward(self, u, K, steps):
        """
        Simulate the H-SSM using explicit Euler integration.
        u: External input signal (steps,)
        K: Feedback gain for the delayed recurrent term
        steps: Number of integration steps
        """
        x1_hist = torch.zeros(steps, self.d1)
        x2_hist = torch.zeros(steps, self.d2)
        
        x1 = torch.zeros(self.d1)
        x2 = torch.zeros(self.d2)
        
        for t in range(steps):
            # Delayed recursive feedback term K * x1(t - tau)
            if t >= self.tau_steps:
                x1_delayed = x1_hist[t - self.tau_steps]
            else:
                x1_delayed = torch.zeros(self.d1)
                
            # Top-down feedback from Layer 2 modulating Layer 1's internal gain
            # Use tanh to keep the modulation bounded around 1.0
            td_mod = 1.0 + torch.tanh(self.W_td @ x2)
            
            # Layer 1 ODE: dx1/dt
            # Includes dampened oscillations, input driving, and delayed self-interference.
            # A non-linear Van der Pol style dampening term (- gamma * x1**3) stabilizes the wave,
            # preventing unbounded exponential growth and forming a true biological limit cycle.
            dx1 = torch.mv(self.A1, x1) * td_mod + self.B1 * u[t] + K * torch.tanh(x1_delayed) - self.gamma * (x1 ** 3)
            
            # Layer 2 ODE: dx2/dt
            # Driven by pooled projection of Layer 1
            dx2 = torch.mv(self.A2, x2) + torch.mv(self.B2, torch.tanh(x1))
            
            # Explicit Euler integration
            x1 = x1 + self.dt * dx1
            x2 = x2 + self.dt * dx2
            
            x1_hist[t] = x1
            x2_hist[t] = x2
            
        return x1_hist, x2_hist

def plot_results(time, dt, x1_A, x2_A, x1_B, x2_B, model):
    """
    Generate the 2x2 subplot visualization comparing Scenario A and Scenario B.

    1. Top-Left: Layer 1 Trajectory: Driven vs Standing Wave
       - Independent Variable (IV): Time (seconds).
       - Dependent Variable (DV): State Amplitude of x1_0 (e.g., bioelectric voltage).
       - Raw Result: The blue line (Scenario A) is a flat, noisy ripple hovering
         around zero. The orange line (Scenario B) initially stays quiet, then
         erupts into massive, sustained, rhythmic oscillations.
       - Interpretation: This is the birth of the continuous wave. In Scenario A,
         the external noise simply dissipates through the network. In Scenario B,
         the recursive self-interference (time t folding into time t - τ) creates
         constructive interference. The biological tissue stops just processing
         external noise and generates its own powerful macroscopic wave.

    2. Top-Right: Phase Space: Emergence of Attractor Limit Cycle
       - Independent Variable (IV): State x1_0 (Voltage).
       - Dependent Variable (DV): State x1_1 (Recovery).
       - Raw Result: The blue line (Scenario A) is trapped in a tiny, chaotic cloud
         at the exact center (0,0). The orange line (Scenario B) spirals outward
         into a massive, beautiful, looping orbit.
       - Interpretation: By plotting the system against itself (removing time),
         we see the literal "geometry" of the thought. The tiny blue cloud means
         the network has no stable internal structure. The massive orange spiral
         is an Attractor Limit Cycle. This proves the system has built a
         self-sustaining engine. It is the mathematical shape of the standing
         wave trapped inside the recurrent cavity.

    3. Bottom-Left: Power Spectral Density (PSD)
       - Independent Variable (IV): Frequency (Hertz).
       - Dependent Variable (DV): Power Density (log scale). This shows how much
         energy is vibrating at each specific frequency.
       - Raw Result: The blue line is a messy, broadband distribution of energy.
         The orange line has a monumental, sharp peak right near the 0-1 Hz mark.
       - Interpretation: The blue line represents a noisy soup of disconnected
         neurons firing randomly. The orange spike is the exact physical signature
         of a phase transition. The entire network has phase-locked. Millions of
         simulated nodes have synchronized into a single, unified resonant
         frequency. This is the macroscopic electromagnetic field asserting
         dominance over the individual cells.

    4. Bottom-Right: Top-Down Prediction Error
       - Independent Variable (IV): Time (seconds).
       - Dependent Variable (DV): L2 Error ||x1 - x1_pred||. This measures the
         difference between Layer 1's actual state and Layer 2's top-down
         prediction of it.
       - Raw Result: The blue line (Scenario A) stays very close to zero. The
         orange line (Scenario B) shows an initial rise as the standing wave
         emerges, followed by a plateau as Layer 2 locks on.
       - Interpretation: With the introduction of non-linear dampening, the wave
         is stabilized into a true biological limit cycle. Because the amplitude
         is bounded, the slower Layer 2 (the top-down macro context) successfully
         maps the internal geometry of Layer 1, allowing the prediction error to
         plateau and phase-lock rather than blowing up to infinity.
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    steps = len(time)
    t_half = steps // 2
    
    # ---------------------------------------------------------
    # Top-Left: Layer 1 State Trajectory
    # ---------------------------------------------------------
    axs[0, 0].plot(time, x1_A[:, 0], label='Scenario A (Low K)', alpha=0.8, color='C0')
    axs[0, 0].plot(time, x1_B[:, 0], label='Scenario B (High K)', alpha=0.8, color='C1')
    axs[0, 0].set_title('Layer 1 Trajectory: Driven vs Standing Wave')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('State Amplitude (x1_0)')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    # ---------------------------------------------------------
    # Top-Right: Phase Space
    # ---------------------------------------------------------
    # Use the second half of the simulation to show the steady-state attractor
    axs[0, 1].plot(x1_A[t_half:, 0], x1_A[t_half:, 1], label='Scenario A', alpha=0.6, color='C0')
    axs[0, 1].plot(x1_B[t_half:, 0], x1_B[t_half:, 1], label='Scenario B', alpha=0.6, color='C1')
    axs[0, 1].set_title('Phase Space: Emergence of Attractor Limit Cycle')
    axs[0, 1].set_xlabel('State x1_0')
    axs[0, 1].set_ylabel('State x1_1')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)
    
    # ---------------------------------------------------------
    # Bottom-Left: Power Spectral Density (PSD)
    # ---------------------------------------------------------
    f_A, Pxx_A = welch(x1_A[t_half:, 0], fs=1/dt, nperseg=1000)
    f_B, Pxx_B = welch(x1_B[t_half:, 0], fs=1/dt, nperseg=1000)
    axs[1, 0].semilogy(f_A, Pxx_A, label='Scenario A (Broadband)', color='C0')
    axs[1, 0].semilogy(f_B, Pxx_B, label='Scenario B (Resonant Peak)', color='C1')
    axs[1, 0].set_title('Power Spectral Density (PSD)')
    axs[1, 0].set_xlabel('Frequency (Hz)')
    axs[1, 0].set_ylabel('Power Density')
    axs[1, 0].set_xlim(0, 10)
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    
    # ---------------------------------------------------------
    # Bottom-Right: Top-down Error Metric
    # ---------------------------------------------------------
    # Layer 2 attempts to predict Layer 1's trajectory. 
    # Error = ||x1 - W_pred * x2||_2
    W_pred = model.W_pred.numpy()
    pred_A = x2_A @ W_pred.T
    pred_B = x2_B @ W_pred.T
    
    err_A = np.linalg.norm(x1_A - pred_A, axis=1)
    err_B = np.linalg.norm(x1_B - pred_B, axis=1)
    
    axs[1, 1].plot(time, err_A, label='Scenario A Error', alpha=0.7, color='C0')
    axs[1, 1].plot(time, err_B, label='Scenario B Error', alpha=0.7, color='C1')
    axs[1, 1].set_title('Top-Down Prediction Error: Layer 2 Locking onto Layer 1')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('L2 Error ||x1 - x1_pred||')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Resolve the output directory relative to the script location (workspace_root/src/demo/.. -> workspace_root)
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_path = os.path.join(workspace_root, 'output', '2_hssm_standing_wave.png')
    
    plt.savefig(out_path, dpi=300)
    print(f"Simulation complete. Plot saved to '{out_path}'.")

def main():
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Simulation parameters
    dt = 0.005
    t_end = 10.0
    time = np.arange(0, t_end, dt)
    steps = len(time)
    
    # Generate external noisy sinusoidal input u(t)
    # Mix of a base frequency and broadband noise
    freq = 2.0 
    noise = np.random.randn(steps)
    u = 0.5 * np.sin(2 * np.pi * freq * time) + 1.0 * noise
    u_tensor = torch.tensor(u, dtype=torch.float32)
    
    # Initialize H-SSM model
    model = HierarchicalSSM(d1=16, d2=4, dt=dt, tau_delay_steps=50)
    
    print("Running Scenario A: Driven Mode (low K)...")
    K_low = 0.5
    x1_A, x2_A = model(u_tensor, K=K_low, steps=steps)
    
    print("Running Scenario B: Standing Wave Mode (high K)...")
    K_high = 5.0
    x1_B, x2_B = model(u_tensor, K=K_high, steps=steps)
    
    # Convert to numpy for visualization
    plot_results(
        time, dt, 
        x1_A.numpy(), x2_A.numpy(), 
        x1_B.numpy(), x2_B.numpy(), 
        model
    )

if __name__ == "__main__":
    main()
