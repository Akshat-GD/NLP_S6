import os
import re
from typing import Dict

from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer

RAW_DATA_PATH = "data/raw/ag_news"
PROCESSED_DATA_PATH = "data/processed/ag_news"
SPLIT_DATA_DIR = "data/splits"
CSV_OUTPUT_DIR = "data/processed"
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128

L1_MAPPING = {
    0: 0,  # World -> Hard News
    1: 1,  # Sports -> Soft News
    2: 0,  # Business -> Hard News
    3: 1   # SciTech -> Soft News
}


def ensure_directories() -> None:
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(SPLIT_DATA_DIR, exist_ok=True)
    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)


def load_raw_dataset() -> DatasetDict:
    dataset = load_dataset("wangrongsheng/ag_news")
    dataset.save_to_disk(RAW_DATA_PATH)
    print(f"Raw dataset saved to: {RAW_DATA_PATH}")
    dataset = load_from_disk(RAW_DATA_PATH)
    print("Raw dataset loaded successfully.")
    return dataset


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_example(example: Dict) -> Dict:
    cleaned_text = clean_text(example["text"])
    l2_label = example["label"]
    l1_label = L1_MAPPING[l2_label]
    return {
        "clean_text": cleaned_text,
        "l1_label": l1_label,
        "l2_label": l2_label,
    }


def preprocess_dataset(dataset: DatasetDict) -> DatasetDict:
    processed_dataset = dataset.map(preprocess_example)
    print("Preprocessing completed.")
    processed_dataset.save_to_disk(PROCESSED_DATA_PATH)
    print(f"Processed dataset saved to: {PROCESSED_DATA_PATH}")
    return processed_dataset


def split_dataset(dataset: DatasetDict) -> Dict[str, object]:
    train_valid = dataset["train"].train_test_split(test_size=0.1, seed=42)
    return {
        "train": train_valid["train"],
        "val": train_valid["test"],
        "test": dataset["test"],
    }


def prepare_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Tokenizer loaded successfully: {MODEL_NAME}")
    return tokenizer


def tokenize_batch(batch: Dict, tokenizer: AutoTokenizer) -> Dict:
    return tokenizer(
        batch["clean_text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


def set_torch_format(dataset, columns) -> None:
    dataset.set_format(type="torch", columns=columns)


def save_dataset_splits(train_dataset, val_dataset, test_dataset) -> None:
    train_dataset.save_to_disk(os.path.join(SPLIT_DATA_DIR, "train"))
    val_dataset.save_to_disk(os.path.join(SPLIT_DATA_DIR, "val"))
    test_dataset.save_to_disk(os.path.join(SPLIT_DATA_DIR, "test"))
    print("Dataset splits saved successfully.")

    train_df = train_dataset.to_pandas()
    val_df = val_dataset.to_pandas()
    test_df = test_dataset.to_pandas()

    train_df.to_csv(os.path.join(CSV_OUTPUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(CSV_OUTPUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(CSV_OUTPUT_DIR, "test.csv"), index=False)
    print("CSV files saved successfully.")


def main() -> None:
    ensure_directories()
    dataset = load_raw_dataset()
    processed_dataset = preprocess_dataset(dataset)

    splits = split_dataset(processed_dataset)
    train_dataset = splits["train"]
    val_dataset = splits["val"]
    test_dataset = splits["test"]
    print("Train/Validation/Test split completed.")

    tokenizer = prepare_tokenizer()
    train_dataset = train_dataset.map(lambda batch: tokenize_batch(batch, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda batch: tokenize_batch(batch, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda batch: tokenize_batch(batch, tokenizer), batched=True)
    print("Tokenization completed.")

    columns = ["input_ids", "attention_mask", "l1_label", "l2_label"]
    set_torch_format(train_dataset, columns)
    set_torch_format(val_dataset, columns)
    set_torch_format(test_dataset, columns)
    print("PyTorch format applied.")

    save_dataset_splits(train_dataset, val_dataset, test_dataset)

    print("First training example:")
    print(train_dataset[0])


if __name__ == "__main__":
    main()
