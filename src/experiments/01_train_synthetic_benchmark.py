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

    optimizer_baseline = optim.AdamW(model_baseline.parameters(), lr=0.005)
    optimizer_mask_aware = optim.AdamW(model_mask_aware.parameters(), lr=0.005)

    criterion = nn.MSELoss()

    baseline_loss_history = []
    mask_aware_loss_history = []

    # Training Loop
    epochs = 30
    for epoch in range(1, epochs + 1):
        model_baseline.train()
        model_mask_aware.train()

        running_loss_baseline = 0.0
        running_loss_mask_aware = 0.0

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

        avg_loss_baseline = running_loss_baseline / len(dataloader)
        avg_loss_mask_aware = running_loss_mask_aware / len(dataloader)

        baseline_loss_history.append(avg_loss_baseline)
        mask_aware_loss_history.append(avg_loss_mask_aware)

        print(
            f"Epoch {epoch}/{epochs} | Baseline Loss: {avg_loss_baseline:.3f} | Mask-Aware Loss: {avg_loss_mask_aware:.3f}"
        )

    # Evaluation & Plotting
    model_baseline.eval()
    model_mask_aware.eval()

    # Fetch a single batch from the dataset to use as a test sequence
    test_batch = next(iter(dataloader))

    # Take the first item in the batch
    test_x_raw = test_batch["x_raw"][0:1].to(device)
    test_mask = test_batch["mask"][0:1].to(device)
    test_y_true = test_batch["y_true"][0].cpu().numpy()

    with torch.no_grad():
        test_pred_baseline = model_baseline(test_x_raw, test_mask)[0].cpu().numpy()
        test_pred_mask_aware = model_mask_aware(test_x_raw, test_mask)[0].cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Top Subplot
    ax1.plot(baseline_loss_history, "r--", label="Baseline")
    ax1.plot(mask_aware_loss_history, "b-", label="Mask-Aware")
    ax1.set_title("MSE Loss Convergence")
    ax1.set_ylabel("MSE")
    ax1.legend()

    # Bottom Subplot
    ax2.plot(test_y_true, "k-", linewidth=3, label="True Phase")
    ax2.plot(test_pred_baseline, "r--", label="Zero-Padded Baseline")
    ax2.plot(test_pred_mask_aware, "b-", label="Mask-Aware Routing")
    ax2.set_title("Waddington Phase Transition Tracking")
    ax2.legend()

    plt.tight_layout()
    os.makedirs("output/data", exist_ok=True)
    plt.savefig("output/data/02_benchmark_results.png")
    print("Saved plot to output/data/02_benchmark_results.png")


if __name__ == "__main__":
    main()
