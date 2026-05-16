import re
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from .config import Config
from .hierarchy import L2_TO_L1


def clean_text(text):
    """Remove HTML tags, backslashes, and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\\", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_and_prepare_data():
    """
    Download AG News, inject L1 labels, and return train/val/test DataFrames.
    Stratified split on l2_label preserves 4-class balance in validation.
    """
    raw = load_dataset(Config.DATASET_NAME)

    def to_df(split):
        df = pd.DataFrame(raw[split]).rename(columns={"label": "l2_label"})
        df["text"]     = df["text"].apply(clean_text)
        df["l1_label"] = df["l2_label"].map(L2_TO_L1)
        return df.reset_index(drop=True)

    train_full = to_df("train")
    test_df    = to_df("test")

    train_df, val_df = train_test_split(
        train_full,
        test_size=Config.VAL_RATIO,
        stratify=train_full["l2_label"],
        random_state=Config.SEED,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df


class AGNewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.texts     = dataframe["text"].tolist()
        self.l1_labels = dataframe["l1_label"].tolist()
        self.l2_labels = dataframe["l2_label"].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=Config.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc["token_type_ids"].squeeze(0),
            "l1_label":       torch.tensor(self.l1_labels[idx], dtype=torch.long),
            "l2_label":       torch.tensor(self.l2_labels[idx], dtype=torch.long),
        }


def build_dataloaders(train_df, val_df, test_df, tokenizer):
    pin = torch.cuda.is_available()
    train_loader = DataLoader(AGNewsDataset(train_df, tokenizer),
                              batch_size=Config.BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=pin)
    val_loader   = DataLoader(AGNewsDataset(val_df,   tokenizer),
                              batch_size=Config.BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=pin)
    test_loader  = DataLoader(AGNewsDataset(test_df,  tokenizer),
                              batch_size=Config.BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=pin)
    return train_loader, val_loader, test_loader