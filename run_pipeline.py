"""
run_pipeline.py
---------------
Entry point for the Hierarchical News Classification pipeline.

Usage (from the project root):
    python run_pipeline.py
    python run_pipeline.py --infer "Fed raises interest rates again."

Stages:
    1. Seed + directory setup
    2. Data download and preparation
    3. Tokenizer and DataLoaders
    4. Model initialisation
    5. Training with early stopping
    6. Final test evaluation + confusion matrices
    7. Inference demo
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
from transformers import BertTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.dataset import load_and_prepare_data, build_dataloaders
from src.model import HierarchicalBERT
from src.train import Trainer
from src.inference import predict, display


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed=Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Hierarchical News Classifier")
    parser.add_argument("--infer", type=str, default=None,
                        metavar="TEXT",
                        help="Run inference on a single text string "
                             "(requires a saved checkpoint)")
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed()

    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR,     exist_ok=True)

    print(f"Device : {Config.DEVICE}")
    print(f"Model  : {Config.MODEL_NAME}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)

    # ── Inference-only shortcut ────────────────────────────────────────────
    if args.infer:
        if not os.path.exists(Config.BEST_MODEL_PATH):
            print("No checkpoint found. Run training first.")
            sys.exit(1)
        model = HierarchicalBERT()
        ckpt  = torch.load(Config.BEST_MODEL_PATH,
                           map_location=Config.DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.to(Config.DEVICE)
        result = predict(args.infer, model, tokenizer,
                         torch.device(Config.DEVICE), verbose=True)
        display(result)
        return

    # ── Data ──────────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    train_df, val_df, test_df = load_and_prepare_data()
    print(f"  train: {len(train_df):,} | val: {len(val_df):,} | "
          f"test: {len(test_df):,}")

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df, tokenizer
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = HierarchicalBERT()

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = Trainer(model, train_loader, val_loader, test_loader)
    trainer.train()

    # ── Inference demo ─────────────────────────────────────────────────────
    sample_texts = [
        "The Federal Reserve raised interest rates for the third consecutive quarter.",
        "Barcelona wins the Champions League final in a dramatic penalty shootout.",
        "Breakthrough in quantum computing achieved by researchers at MIT.",
        "UN Security Council meets to discuss escalating tensions in the region.",
        "Tesla reports record earnings driven by its new AI software subscription.",
    ]

    print("\n" + "=" * 55)
    print("  INFERENCE DEMO")
    print("=" * 55)
    device = torch.device(Config.DEVICE)
    for text in sample_texts:
        result = predict(text, model, tokenizer, device, verbose=True)
        display(result)


if __name__ == "__main__":
    main()