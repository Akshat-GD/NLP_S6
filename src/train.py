import os
import logging
import random
import time
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import BertTokenizer, get_linear_schedule_with_warmup

try:
    from torch.optim import AdamW
except ImportError:
    from transformers import AdamW

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from .config import Config
from .dataset import build_dataloaders
from .model import HierarchicalBERT
from .loss import HierarchicalLoss
from .evaluate import evaluate, print_classification_reports

logger = logging.getLogger(__name__)


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int = Config.SEED) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── Optimizer and scheduler ────────────────────────────────────────────────────

def build_optimizer_and_scheduler(
    model: HierarchicalBERT,
    train_loader: DataLoader,
    num_epochs: int = Config.NUM_EPOCHS,
    bert_lr: float = Config.BERT_LR,
    head_lr: float = Config.HEAD_LR,
    weight_decay: float = Config.WEIGHT_DECAY,
    warmup_ratio: float = Config.WARMUP_RATIO,
) -> Tuple[torch.optim.Optimizer, object]:
    """
    Build AdamW optimiser with differential learning rates and a linear
    warmup + decay learning rate schedule.

    Differential LRs (spec §6.2):
        BERT layers  → bert_lr (2e-5) — lower to avoid catastrophic forgetting
        Head layers  → head_lr (1e-4) — higher for faster convergence

    Warmup (spec §6.2):
        The first 10 % of training steps use a linearly increasing LR, which
        prevents large gradient updates in the early phase when the model
        hasn't yet adapted to the downstream task distribution.
    """
    param_groups = model.get_param_groups(bert_lr, head_lr)

    optimizer = AdamW(param_groups, weight_decay=weight_decay)

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info(
        "Optimiser: AdamW | BERT LR=%.1e | Head LR=%.1e | "
        "total_steps=%d | warmup_steps=%d",
        bert_lr, head_lr, total_steps, warmup_steps,
    )
    return optimizer, scheduler


# ── Single training epoch ──────────────────────────────────────────────────────

def train_epoch(
    model: HierarchicalBERT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: HierarchicalLoss,
    device: str,
    scaler: Optional[GradScaler] = None,
    writer=None,
    global_step: int = 0,
    log_every: int = Config.LOG_EVERY_N_STEPS,
) -> Tuple[float, int]:
    """
    Run one training epoch.

    Args:
        model       : HierarchicalBERT in train mode.
        loader      : Training DataLoader.
        optimizer   : AdamW optimiser.
        scheduler   : LR scheduler (step-level).
        loss_fn     : HierarchicalLoss.
        device      : 'cuda' or 'cpu'.
        scaler      : GradScaler for AMP (None → standard fp32 training).
        writer      : TensorBoard SummaryWriter (optional).
        global_step : Running step counter (for TensorBoard x-axis).
        log_every   : Log scalars every N steps.

    Returns:
        (avg_loss, final_global_step)
    """
    from tqdm import tqdm

    model.train()
    running_loss = 0.0
    running_l1 = 0.0
    running_l2 = 0.0
    running_cons = 0.0

    pbar = tqdm(loader, desc="  Training", leave=False, dynamic_ncols=True)

    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        l1_labels = batch["l1_label"].to(device, non_blocking=True)
        l2_labels = batch["l2_label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            # ── AMP forward pass ───────────────────────────────────────────────
            with autocast():
                l1_logits, l2_logits = model(input_ids, attention_mask)
                loss_dict = loss_fn.detailed_forward(
                    l1_logits, l2_logits, l1_labels, l2_labels
                )

            scaler.scale(loss_dict["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=Config.GRAD_CLIP
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            # ── Standard fp32 forward pass ────────────────────────────────────
            l1_logits, l2_logits = model(input_ids, attention_mask)
            loss_dict = loss_fn.detailed_forward(
                l1_logits, l2_logits, l1_labels, l2_labels
            )
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=Config.GRAD_CLIP
            )
            optimizer.step()

        scheduler.step()
        global_step += 1

        batch_loss = loss_dict["total"].item()
        running_loss += batch_loss
        running_l1 += loss_dict["l1"].item()
        running_l2 += loss_dict["l2"].item()
        running_cons += loss_dict["consistency"].item()

        # TensorBoard step-level logging
        if writer and global_step % log_every == 0:
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar("train/loss_step", batch_loss, global_step)
            writer.add_scalar("train/lr", current_lr, global_step)

        pbar.set_postfix(loss=f"{batch_loss:.4f}")

    n_batches = len(loader)
    return {
        "loss": running_loss / n_batches,
        "l1_loss": running_l1 / n_batches,
        "l2_loss": running_l2 / n_batches,
        "cons_loss": running_cons / n_batches,
    }, global_step


# ── Full training pipeline ─────────────────────────────────────────────────────

def train(config=Config) -> HierarchicalBERT:
    """
    Orchestrate the full training pipeline:
        1. Seed → 2. Tokenizer → 3. DataLoaders → 4. Model
        5. Optimiser/Scheduler → 6. Loss function
        7. Training loop with early stopping and checkpointing
        8. Final test evaluation on best checkpoint

    Args:
        config : Config class (or instance with the same attributes).

    Returns:
        The best HierarchicalBERT model loaded from checkpoint.
    """
    t0 = time.time()
    config.ensure_dirs()
    set_seed(config.SEED)

    logger.info("Device: %s", config.DEVICE)
    logger.info("AMP enabled: %s", config.USE_AMP)

    # ── TensorBoard ────────────────────────────────────────────────────────────
    writer = None
    if HAS_TENSORBOARD:
        writer = SummaryWriter(log_dir=config.LOG_DIR)
        logger.info("TensorBoard logging to %s", config.LOG_DIR)

    # ── W&B (optional) ────────────────────────────────────────────────────────
    if config.USE_WANDB:
        try:
            import wandb
            wandb.init(
                project=config.WANDB_PROJECT,
                config={k: v for k, v in vars(config).items()
                        if not k.startswith("_") and not callable(v)},
            )
        except ImportError:
            logger.warning("wandb not installed — skipping W&B logging.")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", config.MODEL_NAME)
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(
        tokenizer, batch_size=config.BATCH_SIZE
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = HierarchicalBERT(
        model_name=config.MODEL_NAME,
        num_l1_classes=config.NUM_L1_CLASSES,
        num_l2_classes=config.NUM_L2_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
    ).to(config.DEVICE)

    param_info = model.count_parameters()
    logger.info(
        "Parameters — Total: %s | Trainable: %s | Frozen: %s",
        f"{param_info['total']:,}",
        f"{param_info['trainable']:,}",
        f"{param_info['frozen']:,}",
    )

    # ── Optimiser and scheduler ───────────────────────────────────────────────
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, train_loader, num_epochs=config.NUM_EPOCHS,
        bert_lr=config.BERT_LR, head_lr=config.HEAD_LR,
        weight_decay=config.WEIGHT_DECAY, warmup_ratio=config.WARMUP_RATIO,
    )

    # ── Loss function ─────────────────────────────────────────────────────────
    loss_fn = HierarchicalLoss(
        alpha=config.ALPHA, beta=config.BETA, gamma=config.GAMMA
    )

    # ── AMP Scaler ────────────────────────────────────────────────────────────
    scaler = GradScaler() if config.USE_AMP else None

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1: float = 0.0
    patience_counter: int = 0
    global_step: int = 0

    logger.info("Starting training for up to %d epochs…", config.NUM_EPOCHS)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start = time.time()

        # ── Train ──────────────────────────────────────────────────────────────
        train_stats, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn,
            config.DEVICE, scaler, writer, global_step,
        )

        # ── Validate ───────────────────────────────────────────────────────────
        val_metrics = evaluate(model, val_loader, config.DEVICE)
        elapsed = time.time() - epoch_start

        logger.info(
            "Epoch %d/%d | Train Loss: %.4f (L1=%.4f L2=%.4f Cons=%.4f) | "
            "Val L1-Acc: %.4f | Val L2-Acc: %.4f | Val L2-F1: %.4f | "
            "HCR: %.4f | H-F1: %.4f | %.1fs",
            epoch, config.NUM_EPOCHS,
            train_stats["loss"], train_stats["l1_loss"],
            train_stats["l2_loss"], train_stats["cons_loss"],
            val_metrics["l1_accuracy"], val_metrics["l2_accuracy"],
            val_metrics["l2_macro_f1"], val_metrics["hcr"],
            val_metrics["H-F1"], elapsed,
        )

        # TensorBoard epoch-level logging
        if writer:
            writer.add_scalar("train/loss_epoch", train_stats["loss"], epoch)
            writer.add_scalar("train/l1_loss", train_stats["l1_loss"], epoch)
            writer.add_scalar("train/l2_loss", train_stats["l2_loss"], epoch)
            writer.add_scalar("train/cons_loss", train_stats["cons_loss"], epoch)
            writer.add_scalar("val/l1_accuracy", val_metrics["l1_accuracy"], epoch)
            writer.add_scalar("val/l2_accuracy", val_metrics["l2_accuracy"], epoch)
            writer.add_scalar("val/l1_macro_f1", val_metrics["l1_macro_f1"], epoch)
            writer.add_scalar("val/l2_macro_f1", val_metrics["l2_macro_f1"], epoch)
            writer.add_scalar("val/hcr", val_metrics["hcr"], epoch)
            writer.add_scalar("val/h_f1", val_metrics["H-F1"], epoch)

        # W&B logging
        if config.USE_WANDB:
            try:
                import wandb
                wandb.log({**{f"train/{k}": v for k, v in train_stats.items()},
                           **{f"val/{k}": v for k, v in val_metrics.items()
                              if isinstance(v, float)},
                           "epoch": epoch})
            except Exception:
                pass

        # ── Checkpoint on improvement ──────────────────────────────────────────
        if val_metrics["l2_macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["l2_macro_f1"]
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            patience_counter = 0
            logger.info(
                "  ✓ New best val L2 macro-F1: %.4f — checkpoint saved.",
                best_val_f1,
            )
        else:
            patience_counter += 1
            logger.info(
                "  No improvement (patience %d/%d).",
                patience_counter, config.PATIENCE,
            )

        # ── Early stopping ─────────────────────────────────────────────────────
        if patience_counter >= config.PATIENCE:
            logger.info(
                "Early stopping triggered after epoch %d. "
                "Best val L2 macro-F1: %.4f",
                epoch, best_val_f1,
            )
            break

    # ── Final test evaluation ─────────────────────────────────────────────────
    logger.info("Loading best checkpoint from %s…", config.BEST_MODEL_PATH)
    model.load_state_dict(
        torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    )
    model.to(config.DEVICE)

    logger.info("═" * 60)
    logger.info("FINAL TEST EVALUATION")
    logger.info("═" * 60)
    test_metrics = evaluate(model, test_loader, config.DEVICE)
    print_classification_reports(test_metrics)

    if writer:
        writer.add_scalar("test/l1_accuracy", test_metrics["l1_accuracy"], 0)
        writer.add_scalar("test/l2_accuracy", test_metrics["l2_accuracy"], 0)
        writer.add_scalar("test/l2_macro_f1", test_metrics["l2_macro_f1"], 0)
        writer.add_scalar("test/hcr", test_metrics["hcr"], 0)
        writer.add_scalar("test/h_f1", test_metrics["H-F1"], 0)
        writer.close()

    total_time = (time.time() - t0) / 60
    logger.info("Training complete in %.1f minutes.", total_time)

    return model