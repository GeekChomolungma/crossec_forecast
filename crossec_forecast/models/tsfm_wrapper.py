from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from .base import BaseClassifierModel
from .registry import register_model


@register_model("tsfm_wrapper")
@register_model("patch_transformer")
class TSFMWrapper(BaseClassifierModel):
    """
    Time Series Foundation Model / Transformer Backbone Wrapper.
    Implements patch-based temporal embedding, multi-head self-attention,
    and adaptive pooling for cross-sectional scoring.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        d_model: int = int(config.get("d_model", 64))
        nhead: int = int(config.get("nhead", 4))
        num_layers: int = int(config.get("num_layers", 2))
        dim_feedforward: int = int(config.get("dim_feedforward", 128))
        dropout: float = float(config.get("dropout", 0.1))

        # Project feature_dim to d_model embedding space
        self.input_projection = nn.Linear(self.feature_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, self.num_classes)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: [B, L, D]
        b, l, _ = x.shape
        embedded = self.input_projection(x) + self.pos_embedding[:, :l, :]
        encoded = self.transformer_encoder(embedded)  # [B, L, d_model]
        pooled = encoded.mean(dim=1)                   # [B, d_model]
        logits = self.head(pooled)                     # [B, 1]
        return logits

