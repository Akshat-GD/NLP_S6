from typing import Dict, List, Tuple


# ── Label name mappings ────────────────────────────────────────────────────────

# HuggingFace ag_news label integers (0-indexed, confirmed at load time)
L2_LABEL_NAMES: Dict[int, str] = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "SciTech",
}

L1_LABEL_NAMES: Dict[int, str] = {
    0: "Hard News",
    1: "Soft News",
}

# Reverse mappings
L2_NAME_TO_ID: Dict[str, int] = {v: k for k, v in L2_LABEL_NAMES.items()}
L1_NAME_TO_ID: Dict[str, int] = {v: k for k, v in L1_LABEL_NAMES.items()}


# ── Hierarchical mappings ──────────────────────────────────────────────────────

# L2 integer → L1 integer
# World (0) → Hard (0), Sports (1) → Soft (1), Business (2) → Hard (0), SciTech (3) → Soft (1)
L2_TO_L1: Dict[int, int] = {
    0: 0,  # World      → Hard News
    1: 1,  # Sports     → Soft News
    2: 0,  # Business   → Hard News
    3: 1,  # SciTech    → Soft News
}

# L1 integer → list of valid L2 integers (for consistency enforcement)
L1_TO_VALID_L2: Dict[int, List[int]] = {
    0: [0, 2],  # Hard News → World, Business
    1: [1, 3],  # Soft News → Sports, SciTech
}

# Flat list of (L1_label, L2_label) valid pairs — useful for path-based metrics
VALID_HIERARCHY_PATHS: List[Tuple[int, int]] = [
    (L2_TO_L1[l2], l2) for l2 in sorted(L2_LABEL_NAMES.keys())
]


# ── Utility functions ──────────────────────────────────────────────────────────

def get_l1_from_l2(l2_label: int) -> int:
    """Return the L1 parent label for a given L2 label."""
    if l2_label not in L2_TO_L1:
        raise ValueError(
            f"Unknown L2 label {l2_label}. Valid labels: {list(L2_TO_L1.keys())}"
        )
    return L2_TO_L1[l2_label]


def is_consistent(l1_pred: int, l2_pred: int) -> bool:
    """Return True if l1_pred is the valid parent of l2_pred."""
    return L2_TO_L1.get(l2_pred) == l1_pred


def enforce_consistency(l1_pred: int, l2_logits) -> int:
    """
    Post-processing consistency enforcement.

    If the argmax of l2_logits is inconsistent with l1_pred, mask the
    invalid L2 positions to -inf and re-run argmax over valid positions.

    Args:
        l1_pred   : Predicted L1 class index (0 or 1).
        l2_logits : Raw L2 logits tensor [4] or [1, 4].

    Returns:
        Consistent L2 class index.
    """
    import torch

    logits = l2_logits.flatten()  # ensure 1-D
    l2_pred = torch.argmax(logits).item()

    if is_consistent(l1_pred, l2_pred):
        return l2_pred

    # Mask invalid positions and re-argmax
    masked = logits.clone()
    for idx in range(len(L2_LABEL_NAMES)):
        if idx not in L1_TO_VALID_L2[l1_pred]:
            masked[idx] = float("-inf")
    return torch.argmax(masked).item()


def label_path_str(l1: int, l2: int) -> str:
    """Human-readable path, e.g. 'Hard News → Business'."""
    return f"{L1_LABEL_NAMES.get(l1, '?')} → {L2_LABEL_NAMES.get(l2, '?')}"


def hierarchy_summary() -> None:
    """Print the full hierarchy for inspection."""
    print("\nHierarchical Label Structure")
    print("─" * 42)
    for l1_id, l1_name in L1_LABEL_NAMES.items():
        l2_ids = L1_TO_VALID_L2[l1_id]
        l2_names = [L2_LABEL_NAMES[i] for i in l2_ids]
        print(f"  L1[{l1_id}] {l1_name}")
        for l2_id, l2_name in zip(l2_ids, l2_names):
            print(f"        └─ L2[{l2_id}] {l2_name}")
    print()


if __name__ == "__main__":
    hierarchy_summary()
    print("Valid paths:", VALID_HIERARCHY_PATHS)
    print("is_consistent(0, 2):", is_consistent(0, 2))   # True  (Hard → Business)
    print("is_consistent(0, 1):", is_consistent(0, 1))   # False (Hard → Sports)