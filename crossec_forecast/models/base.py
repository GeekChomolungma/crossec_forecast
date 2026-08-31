from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn as nn


class BaseClassifierModel(nn.Module, ABC):
    """
    Abstract Base Class for all Time-Series / Cross-Sectional Classifier Models.
    Enforces unified forward contract: input [B, L, D] -> output logits [B, 1].
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.seq_len = int(config.get("seq_len", 6))
        self.feature_dim = int(config.get("feature_dim", 24))
        self.num_classes = int(config.get("num_classes", 1))

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, seq_len, feature_dim]
            
        Returns:
            Logits tensor of shape [B, 1] (for binary classification)
        """
        pass

    def predict_proba(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Generate continuous probability scores [0, 1].
        
        Args:
            x: Input tensor of shape [B, seq_len, feature_dim]
            
        Returns:
            Probabilities tensor of shape [B, 1]
        """
        logits = self.forward(x, **kwargs)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        else:
            return torch.softmax(logits, dim=-1)[:, 1:2]

