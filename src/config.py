import os
import torch


class Config:
    # Reproducibility
    SEED                  = 42

    # Model
    MODEL_NAME            = "bert-base-uncased"
    MAX_LENGTH            = 128
    NUM_L1_CLASSES        = 2        # Hard News / Soft News
    NUM_L2_CLASSES        = 4        # World / Sports / Business / Sci/Tech
    HIDDEN_DIM            = 256      # Intermediate dim inside classification heads
    DROPOUT_RATE          = 0.3

    # Training
    BERT_LR               = 2e-5     # Lower LR for pre-trained BERT layers
    HEAD_LR               = 1e-4     # Higher LR for new classification heads
    WEIGHT_DECAY          = 0.01
    BATCH_SIZE            = 32
    NUM_EPOCHS            = 2
    WARMUP_RATIO          = 0.1
    GRAD_CLIP             = 1.0
    LABEL_SMOOTHING       = 0.1
    EARLY_STOP_PATIENCE   = 3

    # Hierarchical loss weights  (alpha*L1 + beta*L2 + gamma*Consistency)
    ALPHA                 = 0.3
    BETA                  = 0.6
    GAMMA                 = 0.1

    # Data
    DATASET_NAME          = "wangrongsheng/ag_news"
    VAL_RATIO             = 0.1

    # Paths
    BASE_DIR              = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CHECKPOINT_DIR        = os.path.join(BASE_DIR, "checkpoints")
    OUTPUT_DIR            = os.path.join(BASE_DIR, "outputs")
    BEST_MODEL_PATH       = os.path.join(BASE_DIR, "checkpoints", "best_model.pt")

    # Device
    DEVICE                = "cuda" if torch.cuda.is_available() else "cpu"