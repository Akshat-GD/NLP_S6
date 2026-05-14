# src/dataset.py
# Handles the full data pipeline:
#   1. Load AG News from Hugging Face Hub
#   2. Clean and normalize raw text
#   3. Add hierarchical L1/L2 labels
#   4. Stratified train/val split
#   5. AGNewsHierarchicalDataset (PyTorch Dataset)
#   6. DataLoader factory

import os
import re
import logging
from typing import Tuple, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from .hierarchy import L2_TO_L1, get_l1_from_l2
from .config import Config

logger = logging.getLogger(__name__)


# ── Text cleaning ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalize raw AG News text before tokenization.

    Design decisions:
    - Strip leading/trailing whitespace.
    - Remove HTML artifacts (rare in AG News but present in some feeds).
    - Collapse multiple whitespace characters to a single space.
    - Do NOT lowercase: bert-base-uncased handles it internally — double
      lowercasing would be redundant and could corrupt the tokenizer.
    - Do NOT remove punctuation: BERT's WordPiece tokenizer handles it
      correctly and punctuation carries syntactic information.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    # Remove HTML/XML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


# ── Data loading ───────────────────────────────────────────────────────────────

def load_ag_news() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pull AG News from Hugging Face Hub and return (train_df, test_df).

    Label mapping verified at load time:
        HuggingFace 0-indexed:  0=World, 1=Sports, 2=Business, 3=SciTech
    We expose these directly as `l2_label` and derive `l1_label` via
    the engineered hierarchy.
    """
    logger.info("Loading AG News from HuggingFace Hub (%s)…", Config.HF_DATASET)
    raw = load_dataset(Config.HF_DATASET)

    # Verify label names at load time
    label_feature = raw["train"].features.get("label")
    if label_feature is not None and hasattr(label_feature, "names"):
        logger.info("HuggingFace label names: %s", label_feature.names)

    train_df = pd.DataFrame(raw["train"])
    test_df = pd.DataFrame(raw["test"])

    # Rename 'label' column → 'l2_label' for clarity
    train_df = train_df.rename(columns={"label": "l2_label"})
    test_df = test_df.rename(columns={"label": "l2_label"})

    # Derive L1 labels
    train_df["l1_label"] = train_df["l2_label"].map(L2_TO_L1)
    test_df["l1_label"] = test_df["l2_label"].map(L2_TO_L1)

    logger.info(
        "Train: %d samples | Test: %d samples", len(train_df), len(test_df)
    )
    logger.info(
        "Train L2 distribution:\n%s",
        train_df["l2_label"].value_counts().to_string(),
    )
    logger.info(
        "Train L1 distribution:\n%s",
        train_df["l1_label"].value_counts().to_string(),
    )
    return train_df, test_df


def create_splits(
    train_df: pd.DataFrame,
    val_ratio: float = Config.VAL_RATIO,
    seed: int = Config.SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split of the training data into train and validation sets.

    Stratification is on `l2_label` (fine-grained), which also guarantees
    L1 balance since L1 is a deterministic function of L2.

    Returns:
        (train_data, val_data) DataFrames with reset indices.
    """
    train_data, val_data = train_test_split(
        train_df,
        test_size=val_ratio,
        stratify=train_df["l2_label"],
        random_state=seed,
    )
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    logger.info(
        "Split → Train: %d | Val: %d", len(train_data), len(val_data)
    )
    return train_data, val_data


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    splits_dir: str = Config.SPLITS_DIR,
) -> None:
    """Persist splits to CSV for reproducibility and offline re-use."""
    os.makedirs(splits_dir, exist_ok=True)
    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test.csv"), index=False)
    logger.info("Splits saved to %s/", splits_dir)


def load_splits(
    splits_dir: str = Config.SPLITS_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-saved CSV splits (avoids re-downloading HuggingFace data)."""
    train_df = pd.read_csv(os.path.join(splits_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(splits_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(splits_dir, "test.csv"))
    logger.info(
        "Loaded splits → Train: %d | Val: %d | Test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class AGNewsHierarchicalDataset(Dataset):
    """
    PyTorch Dataset for the Hierarchical News Classification task.

    Each item returns a dict with:
        input_ids      : [max_length]   LongTensor — BERT token IDs
        attention_mask : [max_length]   LongTensor — 1=real token, 0=padding
        l1_label       : scalar LongTensor — Hard(0) / Soft(1)
        l2_label       : scalar LongTensor — World(0)/Sports(1)/Business(2)/SciTech(3)

    Text is cleaned (clean_text) inside __getitem__ so the cleaning logic
    is applied lazily and consistently regardless of how the DF was created.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: BertTokenizer,
        max_length: int = Config.MAX_LENGTH,
    ) -> None:
        self.texts: list = dataframe["text"].tolist()
        self.l1_labels: list = dataframe["l1_label"].tolist()
        self.l2_labels: list = dataframe["l2_label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = clean_text(self.texts[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",    # Pad all sequences to max_length
            truncation=True,         # Truncate sequences > max_length
            return_tensors="pt",     # Return PyTorch tensors
            return_attention_mask=True,
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),         # [max_len]
            "attention_mask": encoding["attention_mask"].squeeze(0), # [max_len]
            "l1_label": torch.tensor(self.l1_labels[idx], dtype=torch.long),
            "l2_label": torch.tensor(self.l2_labels[idx], dtype=torch.long),
        }


# ── DataLoader factory ─────────────────────────────────────────────────────────

def build_dataloaders(
    tokenizer: BertTokenizer,
    batch_size: int = Config.BATCH_SIZE,
    splits_dir: str = Config.SPLITS_DIR,
    num_workers: int = 4,
    reload_from_hub: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, and test DataLoaders.

    If pre-saved CSVs exist in splits_dir and reload_from_hub is False,
    data is loaded from disk (faster). Otherwise pulls from HuggingFace.

    Args:
        tokenizer       : Instantiated BertTokenizer.
        batch_size      : Samples per batch.
        splits_dir      : Directory to read/write CSV splits.
        num_workers     : DataLoader worker processes.
        reload_from_hub : Force re-download from HuggingFace.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_csv = os.path.join(splits_dir, "train.csv")

    if not reload_from_hub and os.path.exists(train_csv):
        logger.info("Loading splits from disk: %s", splits_dir)
        train_df, val_df, test_df = load_splits(splits_dir)
    else:
        train_df, test_df = load_ag_news()
        train_df, val_df = create_splits(train_df)
        save_splits(train_df, val_df, test_df, splits_dir)

    train_dataset = AGNewsHierarchicalDataset(train_df, tokenizer)
    val_dataset = AGNewsHierarchicalDataset(val_df, tokenizer)
    test_dataset = AGNewsHierarchicalDataset(test_df, tokenizer)

    # pin_memory speeds up GPU data transfers
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,   # Avoids a batch-norm edge case on the last mini-batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batches on eval (no gradient mem)
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    logger.info(
        "DataLoaders built → train batches: %d | val batches: %d | test batches: %d",
        len(train_loader), len(val_loader), len(test_loader),
    )
    return train_loader, val_loader, test_loader
