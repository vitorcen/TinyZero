"""Base model."""

import torch.nn as nn


class BaseModel(nn.Module):
    """Base model."""

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        """Forward pass."""
        raise NotImplementedError

    def initialize_parameters(self):
        """Initialize parameters."""
        raise NotImplementedError
