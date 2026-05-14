import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class HierarchicalLoss(nn.Module):
    """
    Weighted composite loss for hierarchical news classification.

    Args:
        alpha     : Weight for L1 cross-entropy loss.
        beta      : Weight for L2 cross-entropy loss.
        gamma     : Weight for KL-divergence consistency loss.
        l1_smoothing : Label smoothing for L1 CE (0.0 = off).
        l2_smoothing : Label smoothing for L2 CE (0.0 = off).

    Label smoothing (optional, spec §9.1):
        Adding label_smoothing=0.1 to CrossEntropyLoss prevents the model
        from becoming overconfident — can help with L1/L2 overlap cases
        such as Sci/Tech articles that use Business vocabulary.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.6,
        gamma: float = 0.1,
        l1_smoothing: float = 0.0,
        l2_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.l1_ce = nn.CrossEntropyLoss(label_smoothing=l1_smoothing)
        self.l2_ce = nn.CrossEntropyLoss(label_smoothing=l2_smoothing)

        # KLDivLoss expects log-probabilities as input, probabilities as target.
        # reduction='batchmean' divides by batch size → scale-independent.
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

        logger.info(
            "HierarchicalLoss | α=%.2f β=%.2f γ=%.2f | "
            "l1_smooth=%.2f l2_smooth=%.2f",
            alpha, beta, gamma, l1_smoothing, l2_smoothing,
        )

    def consistency_loss(
        self,
        l1_logits: torch.Tensor,
        l2_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL-divergence between the L1 head's predicted distribution and the
        L1 distribution *implied* by the L2 head.

        Implied L1 probabilities:
            P(Hard | x) ≈ P_L2(World | x) + P_L2(Business | x)   [cols 0 + 2]
            P(Soft | x) ≈ P_L2(Sports | x) + P_L2(SciTech | x)   [cols 1 + 3]

        The KL term is computed as KL(L1_probs || implied_L1_probs).
        We detach implied_l1 so gradients only flow back through the L1 head;
        the L2 head receives its gradient exclusively from the L2 CE loss.

        Args:
            l1_logits : [B, 2] — raw L1 scores.
            l2_logits : [B, 4] — raw L2 scores.

        Returns:
            Scalar consistency loss.
        """
        l1_probs = torch.softmax(l1_logits, dim=-1)          # [B, 2]
        l2_probs = torch.softmax(l2_logits, dim=-1)          # [B, 4]

        # Aggregate L2 probs into implied L1 probs
        # Hard News = World (col 0) + Business (col 2)
        # Soft News = Sports (col 1) + SciTech (col 3)
        implied_hard = l2_probs[:, 0] + l2_probs[:, 2]      # [B]
        implied_soft = l2_probs[:, 1] + l2_probs[:, 3]      # [B]
        implied_l1 = torch.stack(
            [implied_hard, implied_soft], dim=-1
        ).detach()                                            # [B, 2], no grad

        # KLDivLoss(input=log_probs, target=probs)
        # Add eps for numerical stability in log
        loss = self.kl_div(
            torch.log(l1_probs + 1e-8),
            implied_l1,
        )
        return loss

    def forward(
        self,
        l1_logits: torch.Tensor,
        l2_logits: torch.Tensor,
        l1_labels: torch.Tensor,
        l2_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the weighted hierarchical loss.

        Args:
            l1_logits : [B, 2]  — raw L1 logits.
            l2_logits : [B, 4]  — raw L2 logits.
            l1_labels : [B]     — ground-truth L1 class indices.
            l2_labels : [B]     — ground-truth L2 class indices.

        Returns:
            Scalar total loss.
        """
        l1_loss = self.l1_ce(l1_logits, l1_labels)
        l2_loss = self.l2_ce(l2_logits, l2_labels)
        cons_loss = self.consistency_loss(l1_logits, l2_logits)

        total = (
            self.alpha * l1_loss
            + self.beta * l2_loss
            + self.gamma * cons_loss
        )
        return total

    def detailed_forward(
        self,
        l1_logits: torch.Tensor,
        l2_logits: torch.Tensor,
        l1_labels: torch.Tensor,
        l2_labels: torch.Tensor,
    ) -> dict:
        """
        Same as forward() but returns all component losses for logging.

        Returns:
            dict with keys: 'total', 'l1', 'l2', 'consistency'
        """
        l1_loss = self.l1_ce(l1_logits, l1_labels)
        l2_loss = self.l2_ce(l2_logits, l2_labels)
        cons_loss = self.consistency_loss(l1_logits, l2_logits)

        total = (
            self.alpha * l1_loss
            + self.beta * l2_loss
            + self.gamma * cons_loss
        )
        return {
            "total": total,
            "l1": l1_loss,
            "l2": l2_loss,
            "consistency": cons_loss,
        }