from typing import Dict, Any
import torch
import torch.nn as nn
from .base import BaseClassifierModel
from .registry import register_model


@register_model("lstm")
class LSTMClassifier(BaseClassifierModel):
    """
    Long Short-Term Memory (LSTM) Sequence Classifier.
    Processes the temporal sequence [B, L, D] through recurrent layers
    with configurable pooling and a classification head.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        hidden_dim: int = int(config.get("hidden_dim", 64))
        num_layers: int = int(config.get("num_layers", 2))
        dropout: float = float(config.get("dropout", 0.2))
        bidirectional: bool = bool(config.get("bidirectional", False))
        self.pooling: str = str(config.get("pooling", "last")).lower()

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)

        if self.pooling == "attention":
            self.attn_layer = nn.Linear(out_dim, 1)

        self.head = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(32, self.num_classes),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: [B, L, D]
        lstm_out, _ = self.lstm(x)  # [B, L, out_dim]

        if self.pooling == "mean":
            pooled = lstm_out.mean(dim=1)
        elif self.pooling == "attention":
            weights = torch.softmax(self.attn_layer(lstm_out), dim=1)  # [B, L, 1]
            pooled = (lstm_out * weights).sum(dim=1)                    # [B, out_dim]
        else:  # "last"
            pooled = lstm_out[:, -1, :]                                 # [B, out_dim]

        logits = self.head(pooled)  # [B, 1]
        return logits

