import os
import torch

class Config:
    """
    All hyperparameters and constants for the Hierarchical News Classifier.

    Design rationale:
    - BERT LR (2e-5): Standard for BERT fine-tuning. Above 5e-5 risks
      catastrophic forgetting of pre-trained representations.
    - Head LR (1e-4): Newly initialized heads need faster convergence.
    - ALPHA/BETA/GAMMA: L2 is the primary task, so β > α. Consistency
      loss (γ) is a soft regularizer — too high and it dominates training.
    - MAX_LENGTH = 128: AG News avg ~40 tokens; 128 covers 99%+ of samples.
    """

    # ── Model ────────────────────────────────────────────────────────────────
    MODEL_NAME: str = "bert-base-uncased"
    MAX_LENGTH: int = 128
    NUM_L1_CLASSES: int = 2    # Hard News, Soft News
    NUM_L2_CLASSES: int = 4    # World, Sports, Business, SciTech
    DROPOUT_RATE: float = 0.3

    # ── Training ─────────────────────────────────────────────────────────────
    BERT_LR: float = 2e-5       # Pre-trained BERT layers
    HEAD_LR: float = 1e-4       # New classification heads
    WEIGHT_DECAY: float = 0.01  # L2 regularisation via AdamW
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 8
    WARMUP_RATIO: float = 0.1   # 10 % of total steps used for LR warm-up
    GRAD_CLIP: float = 1.0      # Max gradient norm (prevents exploding grads)
    PATIENCE: int = 3           # Early stopping patience (epochs)
    VAL_RATIO: float = 0.1      # 10 % of train → validation

    # ── Loss weights ─────────────────────────────────────────────────────────
    ALPHA: float = 0.3   # L1 (Hard/Soft) cross-entropy weight
    BETA: float = 0.6    # L2 (4-class) cross-entropy weight  ← primary task
    GAMMA: float = 0.1   # Consistency KL-divergence weight

    # ── Paths ────────────────────────────────────────────────────────────────
    DATA_DIR: str = "data"
    RAW_DIR: str = os.path.join(DATA_DIR, "raw")
    PROCESSED_DIR: str = os.path.join(DATA_DIR, "processed")
    SPLITS_DIR: str = os.path.join(DATA_DIR, "splits")
    CHECKPOINT_DIR: str = "checkpoints"
    LOG_DIR: str = "logs"
    OUTPUT_DIR: str = "outputs"
    BEST_MODEL_PATH: str = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    # ── Reproducibility ───────────────────────────────────────────────────────
    SEED: int = 42

    # ── Device ───────────────────────────────────────────────────────────────
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Mixed precision (AMP) ─────────────────────────────────────────────────
    USE_AMP: bool = torch.cuda.is_available()  # Only meaningful on GPU

    # ── Dataset source ────────────────────────────────────────────────────────
    HF_DATASET: str = "wangrongsheng/ag_news"

    # ── Logging ───────────────────────────────────────────────────────────────
    USE_WANDB: bool = False      # Set True and configure WANDB_API_KEY to enable
    WANDB_PROJECT: str = "hierarchical-news-clf"
    LOG_EVERY_N_STEPS: int = 50  # TensorBoard scalar logging frequency

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create all required directories if they do not already exist."""
        for directory in [
            cls.RAW_DIR, cls.PROCESSED_DIR, cls.SPLITS_DIR,
            cls.CHECKPOINT_DIR, cls.LOG_DIR, cls.OUTPUT_DIR,
        ]:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def display(cls) -> None:
        """Print all configuration values for logging / reproducibility."""
        print("=" * 60)
        print("  CONFIGURATION")
        print("=" * 60)
        for key, val in vars(cls).items():
            if not key.startswith("_") and not callable(val):
                print(f"  {key:<25} = {val}")
        print("=" * 60)