import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.data.waddington_dataset import SyntheticWaddingtonDataset
from src.models.simulators.waddington_predictor import WaddingtonPredictor
from torch.utils.data import DataLoader, Dataset


# Wrapper for dataset to dynamically change its size for DataLoader compatibility
class DatasetWrapper(Dataset):
    def __init__(self, size: int):
        self.dataset = SyntheticWaddingtonDataset(size=size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def main():
    # Setup Device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Data Preparation
    dataset = DatasetWrapper(size=100)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Model Initialization
    model_baseline = WaddingtonPredictor("baseline").to(device)
    model_mask_aware = WaddingtonPredictor("mask_aware").to(device)
    model_transformer = WaddingtonPredictor("transformer").to(device)

    optimizer_baseline = optim.AdamW(model_baseline.parameters(), lr=0.005)
    optimizer_mask_aware = optim.AdamW(model_mask_aware.parameters(), lr=0.005)
    optimizer_transformer = optim.AdamW(model_transformer.parameters(), lr=0.005)

    criterion = nn.MSELoss()

    baseline_loss_history = []
    mask_aware_loss_history = []
    transformer_loss_history = []

    # Training Loop
    epochs = 30
    for epoch in range(1, epochs + 1):
        model_baseline.train()
        model_mask_aware.train()
        model_transformer.train()

        running_loss_baseline = 0.0
        running_loss_mask_aware = 0.0
        running_loss_transformer = 0.0

        for batch in dataloader:
            x_raw = batch["x_raw"].to(device)
            mask = batch["mask"].to(device)
            y_true = batch["y_true"].to(device)

            # Baseline training
            optimizer_baseline.zero_grad()
            preds_baseline = model_baseline(x_raw, mask)
            loss_baseline = criterion(preds_baseline, y_true)
            loss_baseline.backward()
            optimizer_baseline.step()
            running_loss_baseline += loss_baseline.item()

            # Mask-aware training
            optimizer_mask_aware.zero_grad()
            preds_mask_aware = model_mask_aware(x_raw, mask)
            loss_mask_aware = criterion(preds_mask_aware, y_true)
            loss_mask_aware.backward()
            optimizer_mask_aware.step()
            running_loss_mask_aware += loss_mask_aware.item()

            # Transformer training
            optimizer_transformer.zero_grad()
            preds_transformer = model_transformer(x_raw, mask)
            loss_transformer = criterion(preds_transformer, y_true)
            loss_transformer.backward()
            optimizer_transformer.step()
            running_loss_transformer += loss_transformer.item()

        avg_loss_baseline = running_loss_baseline / len(dataloader)
        avg_loss_mask_aware = running_loss_mask_aware / len(dataloader)
        avg_loss_transformer = running_loss_transformer / len(dataloader)

        baseline_loss_history.append(avg_loss_baseline)
        mask_aware_loss_history.append(avg_loss_mask_aware)
        transformer_loss_history.append(avg_loss_transformer)

        print(
            f"Epoch {epoch}/{epochs} | Baseline Loss: {avg_loss_baseline:.3f} | Mask-Aware Loss: {avg_loss_mask_aware:.3f} | Transformer Loss: {avg_loss_transformer:.3f}"
        )

    # ==========================================
    # EVALUATION: LENGTH EXTRAPOLATION STRESS TEST
    # ==========================================
    print("Running Length Extrapolation Stress Test (seq_len=2000)...")
    model_baseline.eval()
    model_mask_aware.eval()
    model_transformer.eval()

    # Generate an Out-Of-Distribution test sequence (4x longer than training)
    ood_dataset = SyntheticWaddingtonDataset(size=1, seq_len=2000)
    test_batch = ood_dataset[0]

    # Add batch dimension and move to device
    test_x_raw = test_batch["x_raw"].unsqueeze(0).to(device)
    test_mask = test_batch["mask"].unsqueeze(0).to(device)
    test_y_true = test_batch["y_true"].cpu().numpy()

    with torch.no_grad():
        test_pred_baseline = model_baseline(test_x_raw, test_mask)[0].cpu().numpy()
        test_pred_mask_aware = model_mask_aware(test_x_raw, test_mask)[0].cpu().numpy()
        test_pred_transformer = model_transformer(test_x_raw, test_mask)[0].cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top Subplot: Training Loss
    ax1.plot(baseline_loss_history, "r--", linewidth=2, label="Baseline SSM")
    ax1.plot(mask_aware_loss_history, "b-", linewidth=2, label="Mask-Aware SSM")
    ax1.plot(transformer_loss_history, "g:", linewidth=2, label="Transformer")
    ax1.set_title("MSE Loss Convergence (Training on seq_len=500)")
    ax1.set_ylabel("MSE")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    # Bottom Subplot: Length Extrapolation Tracking
    ax2.plot(test_y_true, "k-", linewidth=3, label="True Phase")
    ax2.plot(test_pred_baseline, "r--", linewidth=1.5, alpha=0.8, label="Zero-Padded Baseline")
    ax2.plot(test_pred_transformer, "g:", linewidth=2, label="Causal Transformer")
    ax2.plot(test_pred_mask_aware, "b-", linewidth=2, label="Mask-Aware Routing")

    # Mark the training horizon
    ax2.axvline(x=500, color="grey", linestyle="--", linewidth=2)
    ax2.text(
        510,
        ax2.get_ylim()[1] * 0.9,
        "Training Horizon\n(Length Extrapolation)",
        color="grey",
        fontsize=10,
    )

    ax2.set_title("Waddington Phase Tracking: Out-Of-Distribution Stress Test (seq_len=2000)")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Phase State")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    os.makedirs("output/data", exist_ok=True)
    plt.savefig("output/data/03_extrapolation_results.png")
    print("Saved plot to output/data/03_extrapolation_results.png")


if __name__ == "__main__":
    main()
