import torch
import torch.nn as nn
from transformers import BertModel
from .config import Config


class ClassificationHead(nn.Module):
    """
    Two-layer head: Linear -> GELU -> Dropout -> Linear
    GELU matches BERT's internal activation; two layers give more capacity
    than a single linear projection.
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, Config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.HIDDEN_DIM, num_classes),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class HierarchicalBERT(nn.Module):
    """
    Shared BERT backbone with two parallel classification heads.

    Both heads receive gradients simultaneously through the shared encoder,
    forcing it to produce representations useful at both hierarchy levels.

    Forward returns:
        l1_logits : [batch, 2]  Hard News / Soft News
        l2_logits : [batch, 4]  World / Sports / Business / Sci/Tech
    """
    def __init__(self):
        super().__init__()
        self.bert    = BertModel.from_pretrained(Config.MODEL_NAME)
        hidden_size  = self.bert.config.hidden_size   # 768
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        self.l1_head = ClassificationHead(hidden_size, Config.NUM_L1_CLASSES)
        self.l2_head = ClassificationHead(hidden_size, Config.NUM_L2_CLASSES)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out      = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        cls_repr = self.dropout(out.pooler_output)  # [batch, 768]
        return self.l1_head(cls_repr), self.l2_head(cls_repr)

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable