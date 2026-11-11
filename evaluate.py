"""
evaluate.py - Rigorous Evaluation and Comparative Benchmarking Suite.
Computes Confusion Matrices, Multi-Class ROC-AUC Curves, Clinical Metrics,
and Grad-CAM Visualizations for test set MRI scans.
"""

import argparse
import json
import os
from pathlib import Path
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.dataset import get_data_loaders, get_transforms, NORM_MEAN, NORM_STD
from src.models.custom_cnn import AlzheimerCustomCNN
from src.models.transfer_model import AlzheimerResNet
from src.explainability import GradCAM, overlay_cam_on_image
from src.utils import (
    plot_confusion_matrix,
    plot_multiclass_roc,
    plot_model_comparison,
    generate_evaluation_summary
)

def load_checkpoint(model_type, checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    if model_type == "custom_cnn":
        model = AlzheimerCustomCNN(num_classes=num_classes)
    elif model_type == "resnet18":
        model = AlzheimerResNet(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names

def run_test_evaluation(model, test_loader, device="cpu"):
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_targets), np.array(all_preds), np.array(all_probs)

def generate_gradcam_gallery(model, class_names, model_name, save_dir="outputs/figures", device="cpu", file_prefix=None):
    """
    Selects one representative test sample per class and generates a side-by-side
    comparison of Original MRI vs Grad-CAM Attention Heatmap.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    samples_dir = Path("data/samples")
    if not samples_dir.exists():
        return

    cam_engine = GradCAM(model, model.get_cam_target_layer())
    _, eval_transform = get_transforms(img_size=(128, 128))

    fig, axes = plt.subplots(len(class_names), 3, figsize=(12, 3.5 * len(class_names)))

    for idx, cname in enumerate(class_names):
        sample_file = samples_dir / f"{cname}_sample_1.jpg"
        if not sample_file.exists():
            continue

        orig_pil = Image.open(sample_file).convert("RGB")
        orig_np = np.array(orig_pil.resize((128, 128)))

        input_tensor = eval_transform(orig_pil).unsqueeze(0).to(device)

        with torch.enable_grad():
            heatmap, pred_class = cam_engine.generate_heatmap(input_tensor, target_class=idx)

        overlay, _ = overlay_cam_on_image(orig_np, heatmap, alpha=0.55)

        # Plot Original
        axes[idx, 0].imshow(orig_np)
        axes[idx, 0].set_title(f"Original MRI: {cname}", fontsize=11, fontweight="bold")
        axes[idx, 0].axis("off")

        # Plot Heatmap alone
        im_heat = axes[idx, 1].imshow(heatmap, cmap="jet")
        axes[idx, 1].set_title("Grad-CAM Activation Map", fontsize=11)
        axes[idx, 1].axis("off")

        # Plot Superimposed
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title(f"Clinical Overlay ({class_names[pred_class]})", fontsize=11, fontweight="bold")
        axes[idx, 2].axis("off")

    plt.tight_layout()
    slug = file_prefix if file_prefix else model_name.lower().replace(' ', '_')
    gallery_file = save_path / f"{slug}_gradcam_gallery.png"
    plt.savefig(gallery_file, dpi=300)
    plt.close()
    print(f"Saved Grad-CAM explainability gallery to {gallery_file}")

def evaluate_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating models on Test Set (1,280 samples) using device: {device}...")

    # Load test dataloader
    _, _, test_loader, class_names = get_data_loaders(
        data_dir="data/processed",
        batch_size=32
    )

    models_dir = Path("outputs/models")
    model_configs = [
        ("Custom CNN", "custom_cnn", models_dir / "custom_cnn_best.pth"),
        ("ResNet-18 (Transfer Learning)", "resnet18", models_dir / "resnet18_best.pth")
    ]

    all_summaries = {}

    for display_name, m_type, ckpt_path in model_configs:
        if not ckpt_path.exists():
            print(f"[WARNING] Checkpoint {ckpt_path} not found. Train the model first via train.py!")
            continue

        print("\n" + "=" * 70)
        print(f"Evaluating Architecture: {display_name}")
        print("=" * 70)

        model, _ = load_checkpoint(m_type, ckpt_path, device=device)
        y_true, y_pred, y_probs = run_test_evaluation(model, test_loader, device=device)

        summary = generate_evaluation_summary(y_true, y_pred, y_probs, class_names, display_name)
        all_summaries[display_name] = summary

        # Print clinical report table
        print(f"Overall Accuracy:          {summary['accuracy'] * 100:.2f}%")
        print(f"Balanced Accuracy:         {summary['balanced_accuracy'] * 100:.2f}%")
        print(f"Macro F1-Score:            {summary['macro_f1'] * 100:.2f}%")
        print(f"Weighted F1-Score:         {summary['weighted_f1'] * 100:.2f}%\n")
        print("Detailed Clinical Classification Report:")
        print("-" * 65)
        print(f"{'Class':<22} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'Support':<8}")
        print("-" * 65)
        for cname in class_names:
            c_metrics = summary["classification_report"][cname]
            print(f"{cname:<22} | {c_metrics['precision']*100:6.2f}%   | {c_metrics['recall']*100:6.2f}%   | {c_metrics['f1-score']*100:6.2f}%   | {int(c_metrics['support']):<8}")
        print("-" * 65)

        # Generate plots with consistent file prefixes
        plot_confusion_matrix(y_true, y_pred, class_names, model_name=display_name, file_prefix=m_type)
        plot_multiclass_roc(y_true, y_probs, class_names, model_name=display_name, file_prefix=m_type)
        generate_gradcam_gallery(model, class_names, model_name=display_name, device=device, file_prefix=m_type)

    # If both models evaluated, generate comparative benchmark chart
    if len(all_summaries) > 1:
        plot_model_comparison(all_summaries)

    # Save complete evaluation JSON
    report_file = Path("outputs/evaluation_report.json")
    with open(report_file, "w") as f:
        json.dump(all_summaries, f, indent=4)
    print(f"\nSaved comprehensive evaluation report to {report_file.resolve()}")

if __name__ == "__main__":
    evaluate_models()
