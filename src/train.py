import os
import time
import torch
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup
from .config import Config
from .loss import HierarchicalLoss
from .evaluate import Evaluator


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.device       = torch.device(Config.DEVICE)
        self.model.to(self.device)

        self.loss_fn  = HierarchicalLoss()
        self.evaluator = Evaluator()

        total_steps   = len(train_loader) * Config.NUM_EPOCHS
        warmup_steps  = int(total_steps * Config.WARMUP_RATIO)

        # Differential LRs: lower for BERT backbone, higher for new heads
        self.optimizer = torch.optim.AdamW([
            {"params": model.bert.parameters(),    "lr": Config.BERT_LR},
            {"params": model.l1_head.parameters(), "lr": Config.HEAD_LR},
            {"params": model.l2_head.parameters(), "lr": Config.HEAD_LR},
        ], weight_decay=Config.WEIGHT_DECAY)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.best_val_f1      = 0.0
        self.patience_counter = 0

    def _train_epoch(self, epoch):
        self.model.train()
        totals = {"total": 0.0, "l1_loss": 0.0, "l2_loss": 0.0, "cons_loss": 0.0}

        pbar = tqdm(self.train_loader,
                    desc=f"Epoch {epoch}/{Config.NUM_EPOCHS} [train]",
                    leave=False)

        for batch in pbar:
            ids   = batch["input_ids"].to(self.device)
            mask  = batch["attention_mask"].to(self.device)
            ttype = batch["token_type_ids"].to(self.device)
            l1_y  = batch["l1_label"].to(self.device)
            l2_y  = batch["l2_label"].to(self.device)

            self.optimizer.zero_grad()
            l1_logits, l2_logits = self.model(ids, mask, ttype)
            losses = self.loss_fn(l1_logits, l2_logits, l1_y, l2_y)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)
            self.optimizer.step()
            self.scheduler.step()

            for k in totals:
                totals[k] += losses[k].item()
            pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

        n = len(self.train_loader)
        return {k: v / n for k, v in totals.items()}

    def train(self):
        total, trainable = self.model.count_parameters()
        print(f"\nModel — total params: {total:,} | trainable: {trainable:,}")
        print(f"Device: {Config.DEVICE}\n")
        print("=" * 65)

        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        start = time.time()

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            t0 = time.time()

            train_losses = self._train_epoch(epoch)
            val_metrics  = self.evaluator.evaluate(self.model, self.val_loader,
                                                   self.device)

            print(
                f"Epoch {epoch:>2}/{Config.NUM_EPOCHS} [{time.time()-t0:.0f}s] | "
                f"Loss {train_losses['total']:.4f} "
                f"(L1={train_losses['l1_loss']:.4f} "
                f"L2={train_losses['l2_loss']:.4f} "
                f"Cons={train_losses['cons_loss']:.4f}) | "
                f"Val L2-Acc {val_metrics['l2_accuracy']:.4f} | "
                f"Val L2-F1 {val_metrics['l2_macro_f1']:.4f} | "
                f"HCR {val_metrics['hcr']:.4f}"
            )

            # Checkpoint on improvement
            if val_metrics["l2_macro_f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["l2_macro_f1"]
                self.patience_counter = 0
                torch.save({"epoch": epoch,
                            "model_state": self.model.state_dict(),
                            "best_val_f1": self.best_val_f1},
                           Config.BEST_MODEL_PATH)
                print(f"  -> Checkpoint saved (val L2 F1 = {self.best_val_f1:.4f})")
            else:
                self.patience_counter += 1
                print(f"  -> No improvement ({self.patience_counter}/{Config.EARLY_STOP_PATIENCE})")

            if self.patience_counter >= Config.EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

        print(f"\nTraining done in {(time.time()-start)/60:.1f} min | "
              f"Best val L2 F1: {self.best_val_f1:.4f}")

        # Load best weights and run final test evaluation
        ckpt = torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"\n{'='*65}\nFINAL TEST EVALUATION\n{'='*65}")
        test_metrics = self.evaluator.evaluate(self.model, self.test_loader,
                                               self.device,
                                               save_confusion_matrix=True)
        self.evaluator.print_summary(test_metrics)
