"""PPO Critic model."""

import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoConfig

from verl.models.base.base import BaseModel


class DataParallelPPOCritic(BaseModel):
    """PPO Critic model."""

    def __init__(self, config, module=None, optimizer=None, tokenizer=None, rank=0, torch_dtype=torch.float32):
        super().__init__()
        self.config = config
        self.module = module
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.rank = rank

        if module is None:
            # Initialize model
            model_config = AutoConfig.from_pretrained(
                config.model.path,
                num_labels=1,
                trust_remote_code=True,
            )
            self.model = AutoModelForTokenClassification.from_pretrained(
                config.model.path,
                config=model_config,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
            self.score = nn.Linear(self.model.config.hidden_size, 1)

            # Initialize parameters
            self.initialize_parameters()
        else:
            self.model = module

    def initialize_parameters(self):
        """Initialize parameters."""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, input_ids, attention_mask=None):
        """Forward pass."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_state = outputs.hidden_states[-1]
        values = self.score(last_hidden_state)
        return values
