"""
train.py - Robust Training Pipeline for Alzheimer's Brain MRI Classification.
Trains Custom CNN and Pretrained ResNet-18 with Class-Weighted CrossEntropyLoss,
Learning Rate Scheduling, and Early Stopping.
"""

import argparse
import os
import tempfile
import time
from pathlib import Path

# Suppress Matplotlib writable directory warning
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Utilize all available CPU threads
if not torch.cuda.is_available() and os.cpu_count():
    torch.set_num_threads(os.cpu_count())

from src.dataset import get_data_loaders, compute_class_weights
from src.models.custom_cnn import AlzheimerCustomCNN
from src.models.transfer_model import AlzheimerResNet
from src.utils import plot_training_history

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc

def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc

def train_model(
    model_type="custom_cnn",
    epochs=10,
    batch_size=32,
    learning_rate=1e-3,
    img_size=(128, 128),
    patience=4,
    device="cpu"
):
    print("=" * 70)
    print(f"Starting Training: {model_type.upper()} on Device: {device}")
    print("=" * 70)

    # 1. Load Data
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        data_dir="data/processed",
        img_size=img_size,
        batch_size=batch_size
    )

    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # 2. Compute Class Weights for Imbalanced Loss
    class_weights, counts = compute_class_weights(train_loader.dataset, device=device)
    print(f"Train Class Distribution: {dict(zip(class_names, counts))}")
    print(f"Calculated Inverse Class Weights: {np.round(class_weights.cpu().numpy(), 3)}")

    # 3. Instantiate Architecture
    if model_type == "custom_cnn":
        model = AlzheimerCustomCNN(num_classes=num_classes)
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif model_type == "resnet18":
        model = AlzheimerResNet(num_classes=num_classes, pretrained=True, freeze_backbone=False)
        # Differential learning rates: smaller for backbone, larger for new classifier head
        backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
        head_params = [p for n, p in model.named_parameters() if "fc" in n]
        optimizer = AdamW([
            {"params": backbone_params, "lr": learning_rate * 0.1},
            {"params": head_params, "lr": learning_rate}
        ], weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # Output directory setup
    models_dir = Path("outputs/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / f"{model_type}_best.pth"

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    early_stop_counter = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        ep_duration = time.time() - ep_start
        lr_current = optimizer.param_groups[-1]["lr"]

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {lr_current:.6f}"
        )

        # Checkpoint Best Model based on Validation Accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "model_type": model_type,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, best_model_path)
            print(f"  --> Best checkpoint saved to {best_model_path} (Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"\n[INFO] Early stopping triggered at epoch {epoch} (No improvement for {patience} epochs).")
                break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    # Plot training curves
    plot_training_history(history, model_name=f"{model_type.upper()}")

    return best_model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Alzheimer MRI Classifier")
    parser.add_argument("--model", type=str, default="custom_cnn", choices=["custom_cnn", "resnet18", "all"],
                        help="Model architecture to train ('custom_cnn', 'resnet18', or 'all')")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model == "all":
        print(">>> Training Custom CNN Baseline...")
        train_model("custom_cnn", epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, device=device)
        print("\n>>> Training Pretrained ResNet-18...")
        train_model("resnet18", epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr * 0.5, device=device)
    else:
        train_model(args.model, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, device=device)
