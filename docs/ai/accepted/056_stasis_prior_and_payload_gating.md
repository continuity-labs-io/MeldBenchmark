Role Instruction
You are an expert PyTorch ML Engineer. We must implement a "Dual-Lock Stasis" fix to prevent our Mask-Aware SSM from integrating ghost noise, and then execute the true Length Extrapolation benchmark.

Task 1: Update the Fusion Encoder (Stasis Prior)
Modify src/models/encoders/fusion.py. Add a custom initialization to BiologicalCartridgeFusion.
Update the __init__ method so it ends with these exact two lines:

```python
# THE STASIS PRIOR: Default closed (-3), opens when active (+6)
        torch.nn.init.constant_(self.W_gate.bias, -3.0)
        torch.nn.init.constant_(self.W_gate.weight, 6.0 / n_modalities)
```

Task 2: Update the Mask-Aware SSM (Explicit Payload Gating)
Modify src/models/ssm/mask_aware_ssm.py. In the forward loop, we must explicitly gate the input payload so offline sensors inject absolutely zero noise into the hidden state.
Change the inner loop to look exactly like this:
```python
dt_base = torch.nn.functional.softplus(self.dt_proj(x_t))
            dt_gated = dt_base * g_t + 1e-8

            # EXPLICIT INPUT GATING: Block offline sensors from adding ghost noise to the state
            B = self.B_proj(x_t) * g_t
            
            A_bar = torch.exp(A * dt_gated)
            B_bar = (A_bar - 1.0) / (A - 1e-8) * B

            h_prev = A_bar * h_prev + B_bar
            hidden_states.append(h_prev)
```

Task 3: Create the Extrapolation Script
Overwrite src/experiments/02_extrapolation_benchmark.py with the following complete script. (This updates DatasetWrapper to properly accept seq_len so the extrapolation actually runs on 2000 steps!)
```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.data.waddington_dataset import SyntheticWaddingtonDataset
from src.models.simulators.waddington_predictor import WaddingtonPredictor
from torch.utils.data import DataLoader, Dataset

class DatasetWrapper(Dataset):
    def __init__(self, size: int, seq_len: int = 500):
        self.dataset = SyntheticWaddingtonDataset(size=size, seq_len=seq_len)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[idx]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # TRAINING DATA (seq_len = 500)
    train_dataset = DatasetWrapper(size=100, seq_len=500)
    dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    model_baseline = WaddingtonPredictor("baseline").to(device)
    model_mask_aware = WaddingtonPredictor("mask_aware").to(device)
    model_transformer = WaddingtonPredictor("transformer").to(device)

    opt_base = optim.AdamW(model_baseline.parameters(), lr=0.005)
    opt_mask = optim.AdamW(model_mask_aware.parameters(), lr=0.005)
    opt_trans = optim.AdamW(model_transformer.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    epochs = 40
    print("Training models on seq_len=500...")
    for epoch in range(1, epochs + 1):
        model_baseline.train()
        model_mask_aware.train()
        model_transformer.train()

        for batch in dataloader:
            x_raw, mask, y_true = batch["x_raw"].to(device), batch["mask"].to(device), batch["y_true"].to(device)

            opt_base.zero_grad()
            loss_base = criterion(model_baseline(x_raw, mask), y_true)
            loss_base.backward()
            opt_base.step()

            opt_mask.zero_grad()
            loss_mask = criterion(model_mask_aware(x_raw, mask), y_true)
            loss_mask.backward()
            opt_mask.step()

            opt_trans.zero_grad()
            loss_trans = criterion(model_transformer(x_raw, mask), y_true)
            loss_trans.backward()
            opt_trans.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Base: {loss_base.item():.3f} | Mask: {loss_mask.item():.3f} | Trans: {loss_trans.item():.3f}")

    # ==========================================
    # EVALUATION: LENGTH EXTRAPOLATION STRESS TEST
    # ==========================================
    print("\nRunning Length Extrapolation Stress Test (seq_len=2000)...")
    model_baseline.eval()
    model_mask_aware.eval()
    model_transformer.eval()

    # OOD DATA (seq_len = 2000)
    test_dataset = DatasetWrapper(size=1, seq_len=2000)
    test_batch = next(iter(DataLoader(test_dataset, batch_size=1)))

    test_x_raw = test_batch["x_raw"].to(device)
    test_mask = test_batch["mask"].to(device)
    test_y_true = test_batch["y_true"][0].cpu().numpy()

    with torch.no_grad():
        test_pred_baseline = model_baseline(test_x_raw, test_mask)[0].cpu().numpy()
        test_pred_mask_aware = model_mask_aware(test_x_raw, test_mask)[0].cpu().numpy()
        test_pred_transformer = model_transformer(test_x_raw, test_mask)[0].cpu().numpy()

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(test_y_true, "k-", linewidth=3, label="True Phase")
    plt.plot(test_pred_baseline, "r--", linewidth=1.5, alpha=0.8, label="Zero-Padded Baseline")
    plt.plot(test_pred_transformer, "g:", linewidth=2, label="Causal Transformer")
    plt.plot(test_pred_mask_aware, "b-", linewidth=2, label="Mask-Aware Routing")
    
    plt.axvline(x=500, color='grey', linestyle='--', linewidth=2)
    plt.text(510, 1.5, 'Training Horizon (Extrapolation Begins)', color='grey', fontsize=10)
    
    plt.title("Waddington Phase Tracking: Out-Of-Distribution Stress Test (seq_len=2000)")
    plt.xlabel("Time Step")
    plt.ylabel("Phase State")
    plt.legend(loc='upper left')

    plt.tight_layout()
    os.makedirs("output/data", exist_ok=True)
    plt.savefig("output/data/03_extrapolation_results.png")
    print("Saved plot to output/data/03_extrapolation_results.png")

if __name__ == "__main__":
    main()
```
