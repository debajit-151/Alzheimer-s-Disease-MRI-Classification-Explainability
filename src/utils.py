"""
utils.py - Evaluation metrics, visualization utilities (Matplotlib / Seaborn),
and clinical reporting for Alzheimer's MRI classification.
"""

import json
import os
from pathlib import Path
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support
)

# Set clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

def plot_training_history(history, model_name, save_dir="outputs/figures"):
    """
    Plots training & validation loss and accuracy curves over epochs.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#ff7f0e", linewidth=2)
    axes[0].set_title(f"{model_name}: Training & Validation Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Cross Entropy Loss", fontsize=11)
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Accuracy plot
    axes[1].plot(epochs, history["train_acc"], "o-", label="Train Acc", color="#2ca02c", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "s-", label="Val Acc", color="#d62728", linewidth=2)
    axes[1].set_title(f"{model_name}: Training & Validation Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].legend(loc="lower right", frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig_file = save_path / f"{model_name.lower().replace(' ', '_')}_training_curves.png"
    plt.savefig(fig_file, dpi=300)
    plt.close()
    print(f"Saved training curve to {fig_file}")

def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_dir="outputs/figures", file_prefix=None):
    """
    Plots both raw count and normalized confusion matrices side-by-side.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    cm_raw = confusion_matrix(y_true, y_pred)
    cm_norm = cm_raw.astype("float") / (cm_raw.sum(axis=1)[:, np.newaxis] + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(
        cm_raw,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        cbar=False
    )
    axes[0].set_title(f"{model_name}: Confusion Matrix (Raw Counts)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted Diagnosis", fontsize=11)
    axes[0].set_ylabel("Actual Ground Truth", fontsize=11)
    axes[0].tick_params(axis="x", rotation=30)

    # Normalized percentages
    sns.heatmap(
        cm_norm * 100,
        annot=True,
        fmt=".1f",
        cmap="Purples",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
        cbar=True
    )
    axes[1].set_title(f"{model_name}: Confusion Matrix (Normalized %)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted Diagnosis", fontsize=11)
    axes[1].set_ylabel("Actual Ground Truth", fontsize=11)
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    slug = file_prefix if file_prefix else model_name.lower().replace(' ', '_')
    fig_file = save_path / f"{slug}_confusion_matrix.png"
    plt.savefig(fig_file, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {fig_file}")

def plot_multiclass_roc(y_true, y_probs, class_names, model_name, save_dir="outputs/figures", file_prefix=None):
    """
    Generates One-vs-Rest ROC curves with AUC score for each clinical stage.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    num_classes = len(class_names)
    y_true_onehot = np.eye(num_classes)[y_true]

    plt.figure(figsize=(9, 7))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr, tpr,
            color=colors[i % len(colors)],
            lw=2,
            label=f"{class_names[i]} (AUC = {roc_auc:.3f})"
        )

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance Level (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    plt.title(f"{model_name}: Multi-Class ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    slug = file_prefix if file_prefix else model_name.lower().replace(' ', '_')
    fig_file = save_path / f"{slug}_roc_curves.png"
    plt.savefig(fig_file, dpi=300)
    plt.close()
    print(f"Saved ROC curves to {fig_file}")

def plot_model_comparison(comparison_data, save_dir="outputs/figures"):
    """
    Generates a comparative grouped bar chart benchmarking Custom CNN vs ResNet-18
    across Accuracy, Balanced Accuracy, Macro F1, and Weighted F1.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    metrics = ["Accuracy", "Balanced Acc", "Macro F1", "Weighted F1"]
    models = list(comparison_data.keys())

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#3498db", "#2ecc71"]
    for idx, (m_name, m_metrics) in enumerate(comparison_data.items()):
        vals = [
            m_metrics["accuracy"] * 100,
            m_metrics["balanced_accuracy"] * 100,
            m_metrics["macro_f1"] * 100,
            m_metrics["weighted_f1"] * 100
        ]
        offset = (idx - 0.5) * width
        rects = ax.bar(x + offset, vals, width, label=m_name, color=colors[idx % len(colors)], edgecolor="black", alpha=0.85)

        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Architectural Benchmark: Custom CNN vs Transfer Learning (ResNet-18)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, fontweight="bold")
    ax.set_ylim([0, 115])
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig_file = save_path / "model_comparison_benchmark.png"
    plt.savefig(fig_file, dpi=300)
    plt.close()
    print(f"Saved comparison benchmark chart to {fig_file}")

def generate_evaluation_summary(y_true, y_pred, y_probs, class_names, model_name):
    """
    Computes a comprehensive dictionary of all clinical metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    clf_rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    summary = {
        "model_name": model_name,
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": clf_rep
    }

    return summary
