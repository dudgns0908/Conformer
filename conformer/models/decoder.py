from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor

from conformer.models.attention import MultiHeadAttention


class ConformerDecoder(nn.Module):
    """ Conformer decoder """

    def __init__(
            self,
            num_classes: int,
            max_length: int = 300,
            hidden_size: int = 4,
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
            num_heads: int = 4,
            num_layers: int = 2,
            dropout_p: float = 0.3,
            device: torch.device = 'cpu'
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_length = max_length

        self.embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=hidden_size)
        self.attention = MultiHeadAttention(hidden_size, num_heads=num_heads)
        self.dropout = nn.Dropout(p=dropout_p)
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )


        self.fc = nn.Sequential(
            nn.Linear(hidden_size << 1, hidden_size),
            nn.Tanh(),
            # torch.View(shape=(-1, self.hidden_state_dim), contiguous=True),
            nn.Linear(hidden_size, num_classes),
        )

        self.to(device)

    def forward(
            self,
            encoder_outputs: Tensor,
            target: Optional[Tensor],
            teacher_forcing_ratio: float = 1.0
    ):
        # is_teacher_forcing = np.random.rand() < teacher_forcing_ratio
        # if is_teacher_forcing:


        return self.decoder(encoder_outputs)
