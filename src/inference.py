import torch
import torch.nn.functional as F
from .config import Config
from .hierarchy import L1_LABEL_NAMES, L2_LABEL_NAMES, L1_TO_VALID_L2, is_consistent


@torch.no_grad()
def predict(text, model, tokenizer, device, verbose=False):
    """
    Predict hierarchical labels for a single text string.

    Consistency enforcement: if predicted L1 and L2 are incompatible
    (e.g. L1=Hard but L2=Sports), L2 logits are masked for invalid classes
    and re-argmaxed so the output is always parent-child consistent.

    Returns a dict with l1_label, l2_label, confidences, and (if verbose)
    full probability distributions for both levels.
    """
    model.eval()
    enc = tokenizer(
        text,
        max_length=Config.MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    l1_logits, l2_logits = model(enc["input_ids"],
                                 enc["attention_mask"],
                                 enc["token_type_ids"])

    l1_probs = F.softmax(l1_logits, dim=-1).squeeze(0)  # [2]
    l2_probs = F.softmax(l2_logits, dim=-1).squeeze(0)  # [4]

    l1_pred = torch.argmax(l1_probs).item()
    l2_pred = torch.argmax(l2_probs).item()

    # Consistency enforcement
    if not is_consistent(l1_pred, l2_pred):
        valid  = L1_TO_VALID_L2[l1_pred]
        masked = l2_probs.clone()
        for i in range(len(L2_LABEL_NAMES)):
            if i not in valid:
                masked[i] = float("-inf")
        l2_pred = torch.argmax(masked).item()

    result = {
        "text":          text,
        "l1_label":      L1_LABEL_NAMES[l1_pred],
        "l2_label":      L2_LABEL_NAMES[l2_pred],
        "l1_confidence": round(l1_probs[l1_pred].item(), 4),
        "l2_confidence": round(l2_probs[l2_pred].item(), 4),
    }
    if verbose:
        result["l1_probs"] = {L1_LABEL_NAMES[i]: round(l1_probs[i].item(), 4)
                              for i in range(len(L1_LABEL_NAMES))}
        result["l2_probs"] = {L2_LABEL_NAMES[i]: round(l2_probs[i].item(), 4)
                              for i in range(len(L2_LABEL_NAMES))}
    return result


def display(result):
    """Pretty-print a single prediction result."""
    preview = result["text"][:80] + ("..." if len(result["text"]) > 80 else "")
    print("-" * 55)
    print(f"  Text : {preview}")
    print(f"  L1   : {result['l1_label']:<14} (conf: {result['l1_confidence']:.4f})")
    print(f"  L2   : {result['l2_label']:<14} (conf: {result['l2_confidence']:.4f})")
    if "l1_probs" in result:
        print(f"  L1 probs : {result['l1_probs']}")
        print(f"  L2 probs : {result['l2_probs']}")
    print("-" * 55)