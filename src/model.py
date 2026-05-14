import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

from .config import Config

logger = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    """
    Two-layer MLP classification head.

    Linear(hidden_size → 256) → GELU → Dropout → Linear(256 → num_classes)

    The intermediate dimension of 256 provides a learned projection space
    that is task-specific (L1 vs L2) while the BERT backbone remains shared.
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        dropout_rate: float,
        intermediate_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(intermediate_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HierarchicalBERT(nn.Module):
    """
    BERT encoder with two parallel output heads for hierarchical classification.

    Forward pass:
        input_ids, attention_mask → BERT → [CLS] repr →
            L1 head → l1_logits  [B, 2]
            L2 head → l2_logits  [B, 4]

    Args:
        model_name     : HuggingFace model ID (default: 'bert-base-uncased').
        num_l1_classes : Number of L1 classes (Hard/Soft = 2).
        num_l2_classes : Number of L2 classes (World/Sports/Business/SciTech = 4).
        dropout_rate   : Dropout applied after BERT pooler and inside heads.
        freeze_bert_layers : If > 0, freeze the first N encoder layers to
                             reduce overfitting and speed up training.
                             Useful when VRAM is limited or data is small.
    """

    def __init__(
        self,
        model_name: str = Config.MODEL_NAME,
        num_l1_classes: int = Config.NUM_L1_CLASSES,
        num_l2_classes: int = Config.NUM_L2_CLASSES,
        dropout_rate: float = Config.DROPOUT_RATE,
        freeze_bert_layers: int = 0,
    ) -> None:
        super().__init__()

        # ── BERT backbone ──────────────────────────────────────────────────────
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size: int = self.bert.config.hidden_size  # 768 for bert-base

        # Dropout applied to [CLS] pooler output before both heads
        self.dropout = nn.Dropout(dropout_rate)

        # ── Classification heads ───────────────────────────────────────────────
        self.l1_head = ClassificationHead(
            hidden_size, num_l1_classes, dropout_rate
        )
        self.l2_head = ClassificationHead(
            hidden_size, num_l2_classes, dropout_rate
        )

        # ── Optional layer freezing ────────────────────────────────────────────
        if freeze_bert_layers > 0:
            self._freeze_bert_layers(freeze_bert_layers)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "HierarchicalBERT initialised | "
            "backbone: %s | trainable params: %s",
            model_name, f"{n_params:,}",
        )

    def _freeze_bert_layers(self, n_layers: int) -> None:
        """
        Freeze the embedding layer and the first n_layers transformer blocks.

        This reduces the effective parameter count and can prevent
        catastrophic forgetting when fine-tuning data is scarce.
        The classification heads remain trainable.
        """
        # Freeze embedding layer
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        # Freeze first n_layers encoder blocks
        num_hidden = self.bert.config.num_hidden_layers
        n_layers = min(n_layers, num_hidden)
        for layer in self.bert.encoder.layer[:n_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        logger.info(
            "Froze BERT embedding layer + first %d/%d encoder layers.",
            n_layers, num_hidden,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids      : [B, seq_len] — token IDs from BERT tokenizer.
            attention_mask : [B, seq_len] — 1 for real tokens, 0 for padding.
            token_type_ids : [B, seq_len] — segment IDs (optional; defaults to
                             all-zeros inside BertModel for single-sentence input).

        Returns:
            l1_logits : [B, 2] — raw (un-softmaxed) L1 scores.
            l2_logits : [B, 4] — raw (un-softmaxed) L2 scores.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # pooler_output = tanh(Linear([CLS] hidden state)) — aggregates
        # the full sequence into a single sentence-level vector [B, 768].
        cls_repr = self.dropout(outputs.pooler_output)

        l1_logits = self.l1_head(cls_repr)  # [B, 2]
        l2_logits = self.l2_head(cls_repr)  # [B, 4]

        return l1_logits, l2_logits

    def get_param_groups(self, bert_lr: float, head_lr: float) -> list:
        """
        Return parameter groups with differential learning rates.

        Pre-trained BERT layers use a lower LR (bert_lr) to avoid
        catastrophic forgetting. Newly initialised heads use head_lr
        to achieve faster convergence.
        """
        return [
            {"params": self.bert.parameters(), "lr": bert_lr},
            {"params": self.l1_head.parameters(), "lr": head_lr},
            {"params": self.l2_head.parameters(), "lr": head_lr},
        ]

    def count_parameters(self) -> dict:
        """Return dict with total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


def load_model(
    checkpoint_path: str,
    device: str = Config.DEVICE,
    **kwargs,
) -> HierarchicalBERT:
    """
    Load a saved HierarchicalBERT from a checkpoint file.

    Args:
        checkpoint_path : Path to the .pt file saved by torch.save().
        device          : Target device ('cpu' or 'cuda').
        **kwargs        : Passed to HierarchicalBERT.__init__().

    Returns:
        Model in eval mode on the target device.
    """
    model = HierarchicalBERT(**kwargs)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info("Model loaded from %s (device=%s)", checkpoint_path, device)
    return model