import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Config


class HierarchicalLoss(nn.Module):
    """
    Composite loss:  alpha*CE(L1) + beta*CE(L2) + gamma*KL_Consistency

    Consistency term: KL-divergence between the L1 head's output and the
    L1 distribution implied by summing L2 probabilities per parent group.
      implied P(Hard) = P(World) + P(Business)   [L2 indices 0, 2]
      implied P(Soft) = P(Sports) + P(Sci/Tech)  [L2 indices 1, 3]
    The implied tensor is detached so gradients only update the L1 head.
    """
    def __init__(self):
        super().__init__()
        self.l1_ce = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
        self.l2_ce = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    def forward(self, l1_logits, l2_logits, l1_labels, l2_labels):
        l1_loss = self.l1_ce(l1_logits, l1_labels)
        l2_loss = self.l2_ce(l2_logits, l2_labels)

        # Consistency loss
        l1_probs   = F.softmax(l1_logits, dim=-1)
        l2_probs   = F.softmax(l2_logits, dim=-1)
        implied_l1 = torch.stack([
            l2_probs[:, 0] + l2_probs[:, 2],   # Hard: World + Business
            l2_probs[:, 1] + l2_probs[:, 3],   # Soft: Sports + Sci/Tech
        ], dim=-1).detach()
        cons_loss  = F.kl_div(torch.log(l1_probs + 1e-8),
                              implied_l1, reduction="batchmean")

        total = Config.ALPHA * l1_loss + Config.BETA * l2_loss + Config.GAMMA * cons_loss
        return {
            "total":     total,
            "l1_loss":   l1_loss,
            "l2_loss":   l2_loss,
            "cons_loss": cons_loss,
        }