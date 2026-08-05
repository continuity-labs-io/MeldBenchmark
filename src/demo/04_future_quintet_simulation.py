import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.neocortical_engine import NeocorticalEngine

def main():
    print("Narrating the 5 modalities:")
    print("- Sigma (100D): Morphology")
    print("- Psi (12D): RNA")
    print("- Omega (2D): Voltage")
    print("- Gamma (10D): Epigenetic Drift (highly sparse)")
    print("- Mu (5D): Metabolomic Flux (including ATP)")

    print("Initializing 129-D NeocorticalEngine (d_model=256, d_state=64)...")
    engine = NeocorticalEngine(input_dim=129, d_model=256, d_state=64)
    
    print("Generating mock continuous sequence [1, 500, 129]...")
    seq = torch.randn(1, 500, 129)
    
    # Isolate the final dimension (index 128) as the ATP reserve.
    # Linearly deplete the ATP reserve from 1.0 down to 0.0 at frame 300.
    atp_reserve = torch.linspace(1.0, 0.0, 300)
    # Beyond frame 300, it stays at 0.0
    atp_reserve = torch.cat([atp_reserve, torch.zeros(200)])
    
    seq[0, :, 128] = atp_reserve
    
    print("Simulating Causal Crash at frame 300...")
    # When ATP hits 0.0 at frame 300, simulate a cascading Waddington crash
    # by dropping the Voltage (Omega) and structural (Sigma) dimensions to near-zero variance.
    # Assuming Sigma is 0:100, and Omega is 112:114
    seq[0, 300:, :100] = seq[0, 300:, :100] * 0.01
    seq[0, 300:, 112:114] = seq[0, 300:, 112:114] * 0.01
    
    print("Running forward pass to prove hardware capability...")
    try:
        predicted, reconstructed = engine(seq)
        print("Hardware proof: Forward pass succeeded!")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        
    print("Generating 2D heatmap dashboard...")
    plt.figure(figsize=(10, 6))
    plt.imshow(seq[0].T.numpy(), aspect='auto', cmap='viridis', interpolation='none')
    
    # Add horizontal green dashed line at index 128
    plt.axhline(y=128, color='green', linestyle='dashed', label='ATP Metabolic Reserve')
    
    # Add vertical red dashed line at frame 300
    plt.axvline(x=300, color='red', linestyle='dashed', label='ATP Exhaustion (Waddington Crash)')
    
    plt.colorbar(label='Activation / Value')
    plt.xlabel('Time Frame')
    plt.ylabel('Dimension Index')
    plt.title('129-D Quintet Tensor Simulation')
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/04_future_quintet_simulation.png')
    print("Saved to output/04_future_quintet_simulation.png")

if __name__ == '__main__':
    main()
