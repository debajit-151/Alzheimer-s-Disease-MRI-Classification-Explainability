"""
dataset.py - Data loading, transformation pipelines, and class weight computation for Alzheimer's MRI scans.
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ImageNet normalization standard (effective for both pretrained & custom architectures)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

def get_transforms(img_size=(128, 128)):
    """
    Returns data transformation pipelines.
    - Train pipeline includes data augmentation to prevent overfitting & handle class imbalance.
    - Val/Test pipeline applies deterministic resizing and normalization.
    """
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    return train_transform, val_test_transform

def get_data_loaders(
    data_dir="data/processed",
    img_size=(128, 128),
    batch_size=32,
    num_workers=0
):
    """
    Loads train, val, and test splits from disk and returns PyTorch DataLoaders.
    """
    base = Path(data_dir)
    train_dir = base / "train"
    val_dir = base / "val"
    test_dir = base / "test"

    train_transform, eval_transform = get_transforms(img_size)

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    class_names = train_dataset.classes
    return train_loader, val_loader, test_loader, class_names

def compute_class_weights(train_dataset, device="cpu", smoothing=0.5):
    """
    Computes smoothed class weights to prevent gradient explosion on rare classes:
    weight[c] = (Total_Samples / (Num_Classes * Count[c])) ** smoothing
    """
    targets = [s[1] for s in train_dataset.samples]
    class_counts = np.bincount(targets)
    total_samples = len(targets)
    num_classes = len(class_counts)

    raw_weights = total_samples / (num_classes * class_counts.astype(np.float32))
    # Apply square-root / power smoothing
    smoothed_weights = np.power(raw_weights, smoothing)
    # Normalize weights so mean is 1.0
    weights = smoothed_weights / np.mean(smoothed_weights)
    
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    return weights_tensor, class_counts
