import argparse
import logging
import os
import sys
import time

# ── Make src importable from the project root ─────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.hierarchy import hierarchy_summary


def setup_logging(log_dir: str = Config.LOG_DIR) -> None:
    """Configure root logger to write to both console and a log file."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "pipeline.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ── Stage: setup ──────────────────────────────────────────────────────────────

def stage_setup() -> None:
    """Create directories, display config and hierarchy."""
    logger.info(" STAGE: SETUP")
    Config.ensure_dirs()
    Config.display()
    hierarchy_summary()
    logger.info("Setup complete.")


# ── Stage: data ───────────────────────────────────────────────────────────────

def stage_data(reload: bool = False) -> None:
    """Download AG News, apply hierarchy labels, and save CSV splits."""
    logger.info("STAGE: DATA")

    from src.dataset import (
        load_ag_news, create_splits, save_splits,
        load_splits, AGNewsHierarchicalDataset,
    )
    from transformers import BertTokenizer
    import pandas as pd

    splits_exist = os.path.exists(
        os.path.join(Config.SPLITS_DIR, "train.csv")
    )

    if splits_exist and not reload:
        logger.info("Splits already exist at %s. Use --reload to re-download.",
                    Config.SPLITS_DIR)
        train_df, val_df, test_df = load_splits()
    else:
        train_df, test_df = load_ag_news()
        train_df, val_df = create_splits(train_df)
        save_splits(train_df, val_df, test_df)

    logger.info(
        "Splits ready — Train: %d | Val: %d | Test: %d",
        len(train_df), len(val_df), len(test_df),
    )

    # Sanity check tokenizer
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)
    sample = train_df["text"].iloc[0]
    tokens = tokenizer.tokenize(sample)
    logger.info(
        "Tokenizer sanity check — sample length: %d tokens | text: '%.80s…'",
        len(tokens), sample,
    )
    logger.info("Data stage complete.")


# ── Stage: train ──────────────────────────────────────────────────────────────

def stage_train(amp: bool = Config.USE_AMP) -> None:
    """Run the full training loop and save the best checkpoint."""
    logger.info("STAGE: TRAIN")

    from src.train import train
    # Allow the caller to override AMP via CLI
    Config.USE_AMP = amp
    train(Config)
    logger.info("Training stage complete. Checkpoint: %s", Config.BEST_MODEL_PATH)


# ── Stage: evaluate ───────────────────────────────────────────────────────────

def stage_evaluate() -> None:
    """Load best checkpoint, evaluate on test set, save confusion matrices."""
    logger.info("STAGE: EVALUATE")

    if not os.path.exists(Config.BEST_MODEL_PATH):
        logger.error(
            "Checkpoint not found: %s — run the train stage first.",
            Config.BEST_MODEL_PATH,
        )
        sys.exit(1)

    from transformers import BertTokenizer
    from src.model import load_model
    from src.dataset import build_dataloaders
    from src.evaluate import (
        evaluate, print_classification_reports,
        plot_all_confusion_matrices, error_analysis,
    )

    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)
    _, _, test_loader = build_dataloaders(tokenizer, batch_size=Config.BATCH_SIZE)

    logger.info("Loading checkpoint: %s", Config.BEST_MODEL_PATH)
    model = load_model(Config.BEST_MODEL_PATH, device=Config.DEVICE)

    test_metrics = evaluate(model, test_loader, Config.DEVICE)
    print_classification_reports(test_metrics)
    plot_all_confusion_matrices(test_metrics, output_dir=Config.OUTPUT_DIR)

    # Error analysis on a sample of the test set texts
    from src.dataset import load_splits
    _, _, test_df = load_splits()
    error_analysis(test_df["text"].tolist(), test_metrics, n_samples=5)

    logger.info("Evaluation stage complete. Outputs saved to %s/", Config.OUTPUT_DIR)


# ── Stage: infer ──────────────────────────────────────────────────────────────

def stage_infer(texts=None) -> None:
    """Run inference on a list of sample texts."""
    logger.info("STAGE: INFERENCE")

    if not os.path.exists(Config.BEST_MODEL_PATH):
        logger.error(
            "Checkpoint not found: %s — run the train stage first.",
            Config.BEST_MODEL_PATH,
        )
        sys.exit(1)

    from src.inference import load_predictor, predict, format_prediction

    if texts is None:
        texts = [
            "US Treasury Secretary warns of global recession risks amid ongoing trade tensions with China.",
            "Manchester City clinch Premier League title with 3-1 win over Arsenal at the Etihad.",
            "NASA's James Webb Telescope captures first detailed images of exoplanet atmospheres.",
            "Federal Reserve raises interest rates by 25 basis points to curb inflation.",
            "Tiger Woods announces return to PGA Tour after back surgery recovery.",
            "Silicon Valley startup raises $500M Series D to accelerate AI chip development.",
        ]

    model, tokenizer = load_predictor(Config.BEST_MODEL_PATH, Config.DEVICE)

    print("\n" + "═" * 70)
    print("  HIERARCHICAL NEWS CLASSIFIER — Predictions")
    print("═" * 70)
    
    """print("\n" + "═" * 70)
    print("  HIERARCHICAL NEWS CLASSIFIER — Predictions")
    print("═" * 70)"""
    for text in texts:
        pred = predict(text, model, tokenizer)
        print(format_prediction(text, pred))

    logger.info("Inference stage complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hierarchical News Classification Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=["setup", "data", "train", "evaluate", "infer", "all"],
        default="all",
        help=(
            "Pipeline stage to run:\n"
            "  setup    — initialise directories and display config\n"
            "  data     — download AG News and create splits\n"
            "  train    — fine-tune HierarchicalBERT\n"
            "  evaluate — test-set evaluation + confusion matrices\n"
            "  infer    — run inference demo on sample texts\n"
            "  all      — run all stages in sequence (default)"
        ),
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Force re-download of AG News dataset even if splits exist.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable Automatic Mixed Precision (AMP) even if GPU is available.",
    )
    parser.add_argument(
        "--texts",
        nargs="+",
        metavar="TEXT",
        help="Custom text(s) for the inference stage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    use_amp = Config.USE_AMP and not args.no_amp
    stage = args.stage

    logger.info("=" * 60)
    logger.info("  Hierarchical News Classification Pipeline")
    logger.info("  Stage: %s | Device: %s | AMP: %s", stage, Config.DEVICE, use_amp)
    logger.info("=" * 60)
    
    """logger.info("═" * 60)
    logger.info("  Hierarchical News Classification Pipeline")
    logger.info("  Stage: %s | Device: %s | AMP: %s", stage, Config.DEVICE, use_amp)
    logger.info("═" * 60)"""

    t0 = time.time()

    if stage in ("setup", "all"):
        stage_setup()

    if stage in ("data", "all"):
        stage_data(reload=args.reload)

    if stage in ("train", "all"):
        stage_train(amp=use_amp)

    if stage in ("evaluate", "all"):
        stage_evaluate()

    if stage in ("infer", "all"):
        stage_infer(texts=args.texts)

    elapsed = (time.time() - t0) / 60
    logger.info("Pipeline finished in %.1f minutes.", elapsed)


if __name__ == "__main__":
    main()
