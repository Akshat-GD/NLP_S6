import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .config import Config
from .hierarchy import (
    L1_LABEL_NAMES,
    L2_LABEL_NAMES,
    L2_TO_L1,
    L1_TO_VALID_L2,
)

logger = logging.getLogger(__name__)


# ── Core evaluation loop ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    device: str = Config.DEVICE,
) -> Dict:
    """
    Run the model on a DataLoader and collect all predictions and labels.

    Returns a metrics dict with:
        l1_accuracy, l1_macro_f1,
        l2_accuracy, l2_macro_f1,
        hcr (hierarchical consistency rate),
        h_precision, h_recall, h_f1 (hierarchical F1),
        l1_preds, l2_preds, l1_true, l2_true
    """
    model.eval()

    all_l1_preds: List[int] = []
    all_l1_true: List[int] = []
    all_l2_preds: List[int] = []
    all_l2_true: List[int] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        l1_logits, l2_logits = model(input_ids, attention_mask)

        all_l1_preds.extend(torch.argmax(l1_logits, dim=-1).cpu().tolist())
        all_l1_true.extend(batch["l1_label"].tolist())
        all_l2_preds.extend(torch.argmax(l2_logits, dim=-1).cpu().tolist())
        all_l2_true.extend(batch["l2_label"].tolist())

    metrics = _compute_metrics(
        all_l1_true, all_l1_preds,
        all_l2_true, all_l2_preds,
    )
    metrics["l1_preds"] = all_l1_preds
    metrics["l2_preds"] = all_l2_preds
    metrics["l1_true"] = all_l1_true
    metrics["l2_true"] = all_l2_true
    return metrics


# ── Metric computation ─────────────────────────────────────────────────────────

def _compute_metrics(
    l1_true: List[int],
    l1_preds: List[int],
    l2_true: List[int],
    l2_preds: List[int],
) -> Dict:
    """Compute all standard and hierarchical metrics from prediction lists."""
    # Standard metrics
    l1_acc = accuracy_score(l1_true, l1_preds)
    l2_acc = accuracy_score(l2_true, l2_preds)
    l1_f1 = f1_score(l1_true, l1_preds, average="macro", zero_division=0)
    l2_f1 = f1_score(l2_true, l2_preds, average="macro", zero_division=0)

    # Hierarchical metrics
    hcr = hierarchical_consistency_rate(l1_preds, l2_preds)
    hier_metrics = hierarchical_f1(
        list(zip(l1_true, l2_true)),
        list(zip(l1_preds, l2_preds)),
    )

    return {
        "l1_accuracy": l1_acc,
        "l1_macro_f1": l1_f1,
        "l2_accuracy": l2_acc,
        "l2_macro_f1": l2_f1,
        "hcr": hcr,
        **hier_metrics,
    }


# ── Hierarchical metrics ───────────────────────────────────────────────────────

def hierarchical_consistency_rate(
    l1_preds: List[int],
    l2_preds: List[int],
) -> float:
    """
    Fraction of samples where the predicted L1 is consistent with the
    predicted L2 under the engineered hierarchy.

    Expected value: ~0.99+ after training (consistency loss enforces this).
    """
    consistent = sum(
        1
        for l1, l2 in zip(l1_preds, l2_preds)
        if L2_TO_L1.get(l2) == l1
    )
    return consistent / max(len(l1_preds), 1)


def hierarchical_f1(
    true_paths: List[Tuple[int, int]],
    pred_paths: List[Tuple[int, int]],
) -> Dict[str, float]:
    """
    Set-based Hierarchical Precision, Recall, and F1 (Kiritchenko et al.).

    For each sample:
        true_set = {l1_true, l2_true}
        pred_set = {l1_pred, l2_pred}
        HP = |true_set ∩ pred_set| / |pred_set|
        HR = |true_set ∩ pred_set| / |true_set|

    Note: L1 and L2 label indices exist in separate integer spaces, so
    using the flat integer alone would conflate e.g. L1=0 and L2=0.
    We prefix with the level to create distinct set elements.

    Args:
        true_paths : list of (l1_true, l2_true) tuples.
        pred_paths : list of (l1_pred, l2_pred) tuples.

    Returns:
        dict with 'H-Precision', 'H-Recall', 'H-F1'.
    """
    total_hp = 0.0
    total_hr = 0.0

    for (t_l1, t_l2), (p_l1, p_l2) in zip(true_paths, pred_paths):
        # Prefix labels by level to avoid cross-level collision
        true_set = {f"L1_{t_l1}", f"L2_{t_l2}"}
        pred_set = {f"L1_{p_l1}", f"L2_{p_l2}"}

        common = len(true_set & pred_set)
        hp = common / len(pred_set) if pred_set else 0.0
        hr = common / len(true_set) if true_set else 0.0
        total_hp += hp
        total_hr += hr

    n = max(len(true_paths), 1)
    hp_avg = total_hp / n
    hr_avg = total_hr / n
    hf1 = (2 * hp_avg * hr_avg) / (hp_avg + hr_avg + 1e-8)

    return {
        "H-Precision": hp_avg,
        "H-Recall": hr_avg,
        "H-F1": hf1,
    }


# ── Classification reports ─────────────────────────────────────────────────────

def print_classification_reports(metrics: Dict) -> None:
    """Print human-readable classification reports for L1 and L2."""
    l1_names = [L1_LABEL_NAMES[i] for i in sorted(L1_LABEL_NAMES)]
    l2_names = [L2_LABEL_NAMES[i] for i in sorted(L2_LABEL_NAMES)]

    print("\n" + "═" * 60)
    print("  L1 Classification Report (Hard / Soft News)")
    print("═" * 60)
    print(
        classification_report(
            metrics["l1_true"], metrics["l1_preds"],
            target_names=l1_names, digits=4,
        )
    )

    print("═" * 60)
    print("  L2 Classification Report (4-class)")
    print("═" * 60)
    print(
        classification_report(
            metrics["l2_true"], metrics["l2_preds"],
            target_names=l2_names, digits=4,
        )
    )

    print("═" * 60)
    print("  Hierarchical Metrics")
    print("═" * 60)
    print(f"  Hierarchical Consistency Rate (HCR) : {metrics['hcr']:.4f}")
    print(f"  Hierarchical Precision              : {metrics['H-Precision']:.4f}")
    print(f"  Hierarchical Recall                 : {metrics['H-Recall']:.4f}")
    print(f"  Hierarchical F1 (H-F1)              : {metrics['H-F1']:.4f}")
    print("═" * 60 + "\n")


# ── Confusion matrix plotting ──────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    labels: List[str],
    title: str,
    output_dir: str = Config.OUTPUT_DIR,
    figsize: Tuple[int, int] = (8, 6),
) -> str:
    """
    Plot and save a normalised confusion matrix heatmap.

    Both raw counts (annotation) and normalised percentages (colour) are shown.

    Args:
        y_true     : Ground-truth label list.
        y_pred     : Predicted label list.
        labels     : Class name strings in label-index order.
        title      : Plot title and filename stem.
        output_dir : Directory to save the PNG.
        figsize    : Matplotlib figure size.

    Returns:
        Path to saved PNG.
    """
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

    for ax, data, fmt, suffix in [
        (axes[0], cm, "d", "Counts"),
        (axes[1], cm_norm, ".2%", "Normalised"),
    ]:
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(f"{title} — {suffix}", fontsize=13, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    safe_name = title.replace(" ", "_").replace("/", "-")
    out_path = os.path.join(output_dir, f"{safe_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", out_path)
    return out_path


def plot_all_confusion_matrices(
    metrics: Dict,
    output_dir: str = Config.OUTPUT_DIR,
) -> None:
    """Plot and save L1 and L2 confusion matrices."""
    l1_names = [L1_LABEL_NAMES[i] for i in sorted(L1_LABEL_NAMES)]
    l2_names = [L2_LABEL_NAMES[i] for i in sorted(L2_LABEL_NAMES)]

    plot_confusion_matrix(
        metrics["l1_true"], metrics["l1_preds"],
        labels=l1_names, title="L1 Hard vs Soft Confusion Matrix",
        output_dir=output_dir,
    )
    plot_confusion_matrix(
        metrics["l2_true"], metrics["l2_preds"],
        labels=l2_names, title="L2 4-class Confusion Matrix",
        output_dir=output_dir,
    )


# ── Error analysis ─────────────────────────────────────────────────────────────

def error_analysis(
    texts: List[str],
    metrics: Dict,
    n_samples: int = 10,
) -> None:
    """
    Print a sample of erroneous predictions for manual inspection.

    Focuses on two failure modes (spec §4.11):
        1. L1 correct but L2 wrong — fine-grained confusion within a category.
        2. L1/L2 inconsistency — model predicted contradictory levels.
    """
    import random

    l1_true = metrics["l1_true"]
    l1_preds = metrics["l1_preds"]
    l2_true = metrics["l2_true"]
    l2_preds = metrics["l2_preds"]

    print("\n── Error Analysis ────────────────────────────────────")

    # Case 1: L1 correct, L2 wrong
    case1 = [
        i for i, (lt1, lp1, lt2, lp2)
        in enumerate(zip(l1_true, l1_preds, l2_true, l2_preds))
        if lt1 == lp1 and lt2 != lp2
    ]
    print(f"\n[Case 1] L1 correct but L2 wrong: {len(case1)} samples")
    for i in random.sample(case1, min(n_samples, len(case1))):
        print(f"  Text: {texts[i][:120]}")
        print(
            f"  True: {L1_LABEL_NAMES[l1_true[i]]} → {L2_LABEL_NAMES[l2_true[i]]}"
            f" | Pred: {L1_LABEL_NAMES[l1_preds[i]]} → {L2_LABEL_NAMES[l2_preds[i]]}"
        )
        print()

    # Case 2: Inconsistent predictions (L1/L2 disagree)
    case2 = [
        i for i, (lp1, lp2) in enumerate(zip(l1_preds, l2_preds))
        if L2_TO_L1.get(lp2) != lp1
    ]
    print(f"\n[Case 2] Inconsistent L1/L2 predictions: {len(case2)} samples")
    for i in random.sample(case2, min(n_samples, len(case2))):
        print(f"  Text: {texts[i][:120]}")
        print(
            f"  Pred L1: {L1_LABEL_NAMES[l1_preds[i]]} | "
            f"Pred L2: {L2_LABEL_NAMES[l2_preds[i]]} "
            f"(expected L1: {L1_LABEL_NAMES[L2_TO_L1[l2_preds[i]]]})"
        )
        print()
    print("─" * 54 + "\n")