from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification tasks.
    Addresses class imbalance and focuses training on hard negative/positive examples.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, 1], targets: [B, 1] (0 or 1)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = modulating_factor * bce_loss

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss_fn(
    loss_type: str = "bce",
    gamma: float = 2.0,
    alpha: Optional[float] = None,
    **kwargs
) -> nn.Module:
    """Factory function for loss modules."""
    name = loss_type.lower()
    if name == "focal":
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif name in ["bce", "bce_with_logits", "cross_entropy"]:
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported loss type '{loss_type}'. Choose from ['bce', 'focal'].")

