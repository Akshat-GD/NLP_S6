import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from .config import Config
from .dataset import clean_text
from .hierarchy import (
    L1_LABEL_NAMES,
    L2_LABEL_NAMES,
    L1_TO_VALID_L2,
    enforce_consistency,
    label_path_str,
)
from .model import HierarchicalBERT, load_model

logger = logging.getLogger(__name__)


# ── Single-sample prediction ───────────────────────────────────────────────────

@torch.no_grad()
def predict(
    text: str,
    model: HierarchicalBERT,
    tokenizer: BertTokenizer,
    device: str = Config.DEVICE,
) -> Dict:
    """
    Classify a single raw news text.

    Args:
        text      : Raw input string (headline + snippet).
        model     : Loaded HierarchicalBERT in eval mode.
        tokenizer : Matching BERT tokenizer.
        device    : Target device.

    Returns:
        dict with:
            l1         : L1 class name ('Hard News' or 'Soft News')
            l2         : L2 class name ('World', 'Sports', 'Business', 'SciTech')
            l1_id      : L1 class index (int)
            l2_id      : L2 class index (int)
            l1_confidence : Softmax probability of predicted L1 class (float)
            l2_confidence : Softmax probability of predicted L2 class (float)
            l1_probs   : Full L1 probability distribution (list of 2 floats)
            l2_probs   : Full L2 probability distribution (list of 4 floats)
            path       : Human-readable prediction path string
            consistent : Whether L1 and L2 predictions are consistent (bool)
    """
    model.eval()
    cleaned = clean_text(text)

    encoding = tokenizer(
        cleaned,
        max_length=Config.MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    l1_logits, l2_logits = model(input_ids, attention_mask)

    l1_probs_tensor = F.softmax(l1_logits, dim=-1).squeeze(0)  # [2]
    l2_probs_tensor = F.softmax(l2_logits, dim=-1).squeeze(0)  # [4]

    l1_pred = torch.argmax(l1_probs_tensor).item()
    l2_pred_raw = torch.argmax(l2_probs_tensor).item()

    # ── Consistency enforcement ────────────────────────────────────────────────
    # If l2_pred is not a valid child of l1_pred, mask invalid logits
    # and re-argmax over the valid subset.
    l2_pred = enforce_consistency(l1_pred, l2_logits.squeeze(0))

    return {
        "l1": L1_LABEL_NAMES[l1_pred],
        "l2": L2_LABEL_NAMES[l2_pred],
        "l1_id": l1_pred,
        "l2_id": l2_pred,
        "l1_confidence": l1_probs_tensor[l1_pred].item(),
        "l2_confidence": l2_probs_tensor[l2_pred].item(),
        "l1_probs": l1_probs_tensor.cpu().tolist(),
        "l2_probs": l2_probs_tensor.cpu().tolist(),
        "path": label_path_str(l1_pred, l2_pred),
        "consistent": (l2_pred_raw == l2_pred),
    }


# ── Batch prediction ───────────────────────────────────────────────────────────

@torch.no_grad()
def predict_batch(
    texts: List[str],
    model: HierarchicalBERT,
    tokenizer: BertTokenizer,
    device: str = Config.DEVICE,
    batch_size: int = 64,
) -> List[Dict]:
    """
    Classify a list of raw news texts efficiently using mini-batching.

    Args:
        texts      : List of raw input strings.
        model      : Loaded HierarchicalBERT in eval mode.
        tokenizer  : Matching BERT tokenizer.
        device     : Target device.
        batch_size : Number of texts per forward pass.

    Returns:
        List of prediction dicts (same schema as predict()).
    """
    from tqdm import tqdm

    model.eval()
    results: List[Dict] = []

    cleaned_texts = [clean_text(t) for t in texts]

    for start in tqdm(
        range(0, len(cleaned_texts), batch_size),
        desc="Batch inference",
        leave=False,
    ):
        batch_texts = cleaned_texts[start : start + batch_size]

        encoding = tokenizer(
            batch_texts,
            max_length=Config.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        l1_logits, l2_logits = model(input_ids, attention_mask)

        l1_probs_batch = F.softmax(l1_logits, dim=-1)  # [B, 2]
        l2_probs_batch = F.softmax(l2_logits, dim=-1)  # [B, 4]

        l1_preds = torch.argmax(l1_probs_batch, dim=-1)  # [B]
        l2_preds_raw = torch.argmax(l2_probs_batch, dim=-1)  # [B]

        for i in range(len(batch_texts)):
            l1_pred = l1_preds[i].item()
            l2_pred_raw = l2_preds_raw[i].item()
            l2_pred = enforce_consistency(l1_pred, l2_logits[i])

            results.append({
                "l1": L1_LABEL_NAMES[l1_pred],
                "l2": L2_LABEL_NAMES[l2_pred],
                "l1_id": l1_pred,
                "l2_id": l2_pred,
                "l1_confidence": l1_probs_batch[i, l1_pred].item(),
                "l2_confidence": l2_probs_batch[i, l2_pred].item(),
                "l1_probs": l1_probs_batch[i].cpu().tolist(),
                "l2_probs": l2_probs_batch[i].cpu().tolist(),
                "path": label_path_str(l1_pred, l2_pred),
                "consistent": (l2_pred_raw == l2_pred),
            })

    return results


# ── Pretty-print helper ────────────────────────────────────────────────────────

def format_prediction(text: str, pred: Dict) -> str:
    """Return a formatted string representation of a single prediction."""
    l1_names = list(L1_LABEL_NAMES.values())
    l2_names = list(L2_LABEL_NAMES.values())

    l1_bar = " | ".join(
        f"{n}: {p:.3f}" for n, p in zip(l1_names, pred["l1_probs"])
    )
    l2_bar = " | ".join(
        f"{n}: {p:.3f}" for n, p in zip(l2_names, pred["l2_probs"])
    )

    lines = [
        "─" * 70,
        f"  Text   : {text[:100]}{'…' if len(text) > 100 else ''}",
        f"  Path   : {pred['path']}",
        f"  L1     : {pred['l1']} (conf={pred['l1_confidence']:.3f})",
        f"  L2     : {pred['l2']} (conf={pred['l2_confidence']:.3f})",
        f"  Consis.: {'✓ Yes' if pred['consistent'] else '✗ Corrected by enforcement'}",
        f"  L1 dist: [{l1_bar}]",
        f"  L2 dist: [{l2_bar}]",
        "─" * 70,
    ]
    return "\n".join(lines)


# ── Convenience loader ─────────────────────────────────────────────────────────

def load_predictor(
    checkpoint_path: str = Config.BEST_MODEL_PATH,
    device: str = Config.DEVICE,
):
    """
    Load a trained model and its tokenizer ready for inference.

    Returns:
        (model, tokenizer) tuple.
    """
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)
    model = load_model(checkpoint_path, device=device)
    return model, tokenizer


# ── CLI demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    DEMO_TEXTS = [
        "US Treasury Secretary warns of global recession risks amid ongoing trade tensions with China.",
        "Manchester City clinch Premier League title with 3-1 win over Arsenal at the Etihad Stadium.",
        "NASA's James Webb Space Telescope captures first detailed images of exoplanet atmospheres.",
        "Federal Reserve raises interest rates by 25 basis points in bid to curb inflation.",
        "Tiger Woods announces return to PGA Tour after back surgery recovery.",
        "Silicon Valley startup raises $500M Series D to accelerate AI chip development.",
    ]

    checkpoint = sys.argv[1] if len(sys.argv) > 1 else Config.BEST_MODEL_PATH

    print(f"\nLoading model from: {checkpoint}")
    model, tokenizer = load_predictor(checkpoint)

    print("\n" + "═" * 70)
    print("  HIERARCHICAL NEWS CLASSIFIER — Inference Demo")
    print("═" * 70)

    for text in DEMO_TEXTS:
        pred = predict(text, model, tokenizer)
        print(format_prediction(text, pred))