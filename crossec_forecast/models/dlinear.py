from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseClassifierModel
from .registry import register_model


class SeriesDecomp(nn.Module):
    """Series decomposition block into Trend and Seasonal/Remainder components."""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L, D] -> transpose for 1D pooling: [B, D, L]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1).permute(0, 2, 1)  # [B, D, L_pad]

        trend = self.avg(x_pad).permute(0, 2, 1)  # [B, L, D]
        # Adjust length if slight mismatch
        if trend.shape[1] != x.shape[1]:
            trend = trend[:, :x.shape[1], :]
        seasonal = x - trend
        return seasonal, trend


@register_model("dlinear")
class DLinearClassifier(BaseClassifierModel):
    """
    DLinear Classifier: Decomposes series into Trend and Seasonal components
    and applies linear temporal mappings for classification / directional scoring.
    """

    # Raw forward() output is logits [B, 1]; inherits BaseClassifierModel.to_score
    # (sigmoid -> P(beat median)) and .compute_loss (BCE/Focal on batch["y"]).
    output_kind = "binary_prob"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        kernel_size = int(config.get("kernel_size", 3))
        self.individual = bool(config.get("individual", False))

        self.decomp = SeriesDecomp(kernel_size=kernel_size)

        # Map temporal dimension seq_len -> 1
        if self.individual:
            self.linear_seasonal = nn.ModuleList([
                nn.Linear(self.seq_len, 1) for _ in range(self.feature_dim)
            ])
            self.linear_trend = nn.ModuleList([
                nn.Linear(self.seq_len, 1) for _ in range(self.feature_dim)
            ])
        else:
            self.linear_seasonal = nn.Linear(self.seq_len, 1)
            self.linear_trend = nn.Linear(self.seq_len, 1)

        # Final projection: feature_dim -> num_classes
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.num_classes)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: [B, L, D]
        seasonal_init, trend_init = self.decomp(x)

        # Transpose to [B, D, L] for temporal linear mapping
        seasonal_t = seasonal_init.permute(0, 2, 1)
        trend_t = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_out = []
            trend_out = []
            for i in range(self.feature_dim):
                seasonal_out.append(self.linear_seasonal[i](seasonal_t[:, i:i+1, :]))
                trend_out.append(self.linear_trend[i](trend_t[:, i:i+1, :]))
            seasonal_out = torch.cat(seasonal_out, dim=1)  # [B, D, 1]
            trend_out = torch.cat(trend_out, dim=1)        # [B, D, 1]
        else:
            seasonal_out = self.linear_seasonal(seasonal_t)  # [B, D, 1]
            trend_out = self.linear_trend(trend_t)            # [B, D, 1]

        combined = (seasonal_out + trend_out).squeeze(-1)  # [B, D]
        logits = self.classifier(combined)                # [B, 1]
        return logits

