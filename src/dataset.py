# %%
import os
import re
import pandas as pd
import numpy as np

from datasets import (
    load_dataset,
    load_from_disk
)

from sklearn.model_selection import train_test_split

from transformers import BertTokenizer

# %%
RAW_DATA_PATH = "data/raw/ag_news"

PROCESSED_DATA_PATH = "data/processed/ag_news"

SPLIT_DATA_PATH = "data/splits"

# %%
os.makedirs(RAW_DATA_PATH, exist_ok=True)

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

os.makedirs(SPLIT_DATA_PATH, exist_ok=True)

print("Directories created successfully.")

# %%
dataset = load_dataset("wangrongsheng/ag_news")

print(dataset)

# %%
dataset.save_to_disk(RAW_DATA_PATH)

print(f"Raw dataset saved to: {RAW_DATA_PATH}")

# %%
dataset = load_from_disk(RAW_DATA_PATH)

print("Raw dataset loaded successfully.")

# %%
train_df = pd.DataFrame(dataset["train"])

train_df.head()

# %%
train_df["label"].value_counts()

# %%
L2_LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "SciTech"
}

L1_MAPPING = {
    0: 0,  # World -> Hard News
    1: 1,  # Sports -> Soft News
    2: 0,  # Business -> Hard News
    3: 1   # SciTech -> Soft News
}

L1_LABELS = {
    0: "Hard News",
    1: "Soft News"
}

# %%
def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

# %%
def preprocess(example):

    cleaned_text = clean_text(example["text"])

    l2_label = example["label"]

    l1_label = L1_MAPPING[l2_label]

    return {
        "clean_text": cleaned_text,
        "l1_label": l1_label,
        "l2_label": l2_label
    }

# %%
processed_dataset = dataset.map(preprocess)

print("Preprocessing completed.")

# %%
processed_dataset.save_to_disk(
    PROCESSED_DATA_PATH
)

print(f"Processed dataset saved to: {PROCESSED_DATA_PATH}")

# %%
train_valid = processed_dataset["train"].train_test_split(
    test_size=0.1,
    seed=42
)

train_dataset = train_valid["train"]

val_dataset = train_valid["test"]

test_dataset = processed_dataset["test"]

print("Train/Validation/Test split completed.")

# %%
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

print("Tokenizer loaded successfully.")

# %%
def tokenize(batch):

    return tokenizer(
        batch["clean_text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# %%
train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

val_dataset = val_dataset.map(
    tokenize,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize,
    batched=True
)

print("Tokenization completed.")

# %%
columns = [
    "input_ids",
    "attention_mask",
    "l1_label",
    "l2_label"
]

train_dataset.set_format(
    type="torch",
    columns=columns
)

val_dataset.set_format(
    type="torch",
    columns=columns
)

test_dataset.set_format(
    type="torch",
    columns=columns
)

print("PyTorch format applied.")

# %%
train_dataset.save_to_disk(
    "data/splits/train"
)

val_dataset.save_to_disk(
    "data/splits/val"
)

test_dataset.save_to_disk(
    "data/splits/test"
)

print("Dataset splits saved successfully.")

# %%
pd.DataFrame(train_dataset).to_csv(
    "data/processed/train.csv",
    index=False
)

pd.DataFrame(val_dataset).to_csv(
    "data/processed/val.csv",
    index=False
)

pd.DataFrame(test_dataset).to_csv(
    "data/processed/test.csv",
    index=False
)

print("CSV files saved successfully.")

# %%
print(train_dataset[0])

# %%
for root, dirs, files in os.walk("data"):
    print(root)

# %%



