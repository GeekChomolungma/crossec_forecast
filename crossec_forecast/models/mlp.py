from typing import Dict, Any, List
import torch
import torch.nn as nn
from .base import BaseClassifierModel
from .registry import register_model


@register_model("mlp")
class MLPClassifier(BaseClassifierModel):
    """
    Flattened Multi-Layer Perceptron (MLP) Classifier.
    Flattens the temporal lookback window [B, L, D] into [B, L * D]
    and processes through dense non-linear projection layers.
    """

    # Raw forward() output is logits [B, 1]; inherits BaseClassifierModel.to_score
    # (sigmoid -> P(beat median)) and .compute_loss (BCE/Focal on batch["y"]).
    output_kind = "binary_prob"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        hidden_dims: List[int] = config.get("hidden_dims", [64, 32])
        dropout: float = float(config.get("dropout", 0.2))
        use_norm: bool = bool(config.get("use_norm", True))

        in_dim = self.seq_len * self.feature_dim
        layers: List[nn.Module] = []

        curr_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            if use_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        layers.append(nn.Linear(curr_dim, self.num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: [B, L, D] -> [B, L * D]
        b = x.size(0)
        x_flat = x.reshape(b, -1)
        logits = self.network(x_flat)  # [B, 1]
        return logits

