import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
from .config import Config
from .hierarchy import L1_LABEL_NAMES, L2_LABEL_NAMES, is_consistent


class Evaluator:
    def __init__(self):
        pass

    @torch.no_grad()
    def evaluate(self, model, dataloader, device, save_confusion_matrix=False):
        model.eval()
        l1_preds, l1_trues, l2_preds, l2_trues = [], [], [], []

        for batch in dataloader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            ttype = batch["token_type_ids"].to(device)
            l1_logits, l2_logits = model(ids, mask, ttype)
            l1_preds.extend(torch.argmax(l1_logits, dim=-1).cpu().tolist())
            l2_preds.extend(torch.argmax(l2_logits, dim=-1).cpu().tolist())
            l1_trues.extend(batch["l1_label"].tolist())
            l2_trues.extend(batch["l2_label"].tolist())

        metrics = {}

        # Standard metrics for L1 and L2
        for level, preds, trues, names in [
            ("l1", l1_preds, l1_trues, L1_LABEL_NAMES),
            ("l2", l2_preds, l2_trues, L2_LABEL_NAMES),
        ]:
            target_names = [names[i] for i in sorted(names)]
            acc          = accuracy_score(trues, preds)
            p, r, f, _   = precision_recall_fscore_support(trues, preds,
                                                            average="macro",
                                                            zero_division=0)
            metrics[f"{level}_accuracy"]  = acc
            metrics[f"{level}_precision"] = p
            metrics[f"{level}_recall"]    = r
            metrics[f"{level}_macro_f1"]  = f
            metrics[f"{level}_report"]    = classification_report(
                trues, preds, target_names=target_names, zero_division=0)

        # Hierarchical Consistency Rate (HCR)
        metrics["hcr"] = sum(is_consistent(l1, l2)
                             for l1, l2 in zip(l1_preds, l2_preds)) / len(l1_preds)

        # Hierarchical F1 (H-F1)
        # For each sample, true path = {l1_true, l2_true}, pred path = {l1_pred, l2_pred}.
        # HP = |intersection| / |pred|, HR = |intersection| / |true|
        # L1 ids are prefixed negative to avoid collision with L2 ids in the sets.
        total_hp = total_hr = 0.0
        for l1p, l1t, l2p, l2t in zip(l1_preds, l1_trues, l2_preds, l2_trues):
            true_path = {-(l1t + 1), l2t}
            pred_path = {-(l1p + 1), l2p}
            common     = len(true_path & pred_path)
            total_hp  += common / len(pred_path)
            total_hr  += common / len(true_path)
        hp = total_hp / len(l1_preds)
        hr = total_hr / len(l1_preds)
        metrics["h_precision"] = hp
        metrics["h_recall"]    = hr
        metrics["h_f1"]        = (2 * hp * hr / (hp + hr)) if (hp + hr) > 0 else 0.0

        if save_confusion_matrix:
            os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
            self._save_cm(l1_preds, l1_trues, L1_LABEL_NAMES, "cm_l1.png")
            self._save_cm(l2_preds, l2_trues, L2_LABEL_NAMES, "cm_l2.png")

        return metrics

    def _save_cm(self, preds, trues, label_names, filename):
        labels  = sorted(label_names)
        names   = [label_names[i] for i in labels]
        cm      = confusion_matrix(trues, preds, labels=labels)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, data, fmt, title in zip(
            axes, [cm, cm_norm], ["d", ".2f"], ["Counts", "Normalised"]
        ):
            sns.heatmap(data, annot=True, fmt=fmt,
                        xticklabels=names, yticklabels=names,
                        cmap="Blues", ax=ax, linewidths=0.4)
            ax.set_title(title)
            ax.set_ylabel("True")
            ax.set_xlabel("Predicted")

        plt.tight_layout()
        path = os.path.join(Config.OUTPUT_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix saved -> {path}")

    def print_summary(self, metrics):
        print("\n" + "=" * 55)
        print("  EVALUATION RESULTS")
        print("=" * 55)
        print(f"  L1 Accuracy  : {metrics['l1_accuracy']:.4f}")
        print(f"  L1 Macro-F1  : {metrics['l1_macro_f1']:.4f}")
        print(f"  L2 Accuracy  : {metrics['l2_accuracy']:.4f}")
        print(f"  L2 Macro-F1  : {metrics['l2_macro_f1']:.4f}")
        print("-" * 55)
        print(f"  H-Precision  : {metrics['h_precision']:.4f}")
        print(f"  H-Recall     : {metrics['h_recall']:.4f}")
        print(f"  H-F1         : {metrics['h_f1']:.4f}")
        print(f"  HCR          : {metrics['hcr']:.4f}")
        print("=" * 55)
        print("\n-- L1 Classification Report --")
        print(metrics["l1_report"])
        print("-- L2 Classification Report --")
        print(metrics["l2_report"])