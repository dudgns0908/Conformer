from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor

from conformer.models.attention import MultiHeadAttention


class ConformerDecoder(nn.Module):
    """ Conformer decoder """

    def __init__(
            self,
            vocab_size: int,
            hidden_size: int = 640,
            max_length: int = 300,
            sos_id: int = 1,
            eos_id: int = 2,
            pad_id: int = 0,
            num_heads: int = 4,
            num_layers: int = 2,
            dropout_p: float = 0.3,
            device: torch.device = 'cpu'
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.device = device

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
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
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(
            self,
            encoder_outputs: Tensor,
            targets: Optional[Tensor] = None,
            teacher_forcing_ratio: float = 1.0
    ):
        batch_size = encoder_outputs.size(0)
        # if targets is None:
        #     targets = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
        #     max_length = self.max_length
        #     if self.device != 'cpu':
        #         targets = targets.to(self.device)

        encoder_outputs.view((batch_size, self.vocab_size))
        embedding = self.embedding(encoder_outputs.long())
        rnn_out = self.rnn(embedding)
        # attention = self.attention(embedding, embedding, embedding)


        # is_teacher_forcing = np.random.rand() < teacher_forcing_ratio
        # if is_teacher_forcing:
        #     pass

        return self.decoder(encoder_outputs)
