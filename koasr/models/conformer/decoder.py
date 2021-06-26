from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn, Tensor

from koasr.modules.attention import MultiHeadAttention


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
            num_heads: int = 8,
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

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attention = MultiHeadAttention(hidden_size, num_heads=num_heads)
        self.dropout = nn.Dropout(p=dropout_p)
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout_p,
            bias=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
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

        # Check target
        if targets is None:
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            max_length = self.max_length
            if self.device != 'cpu':
                inputs = inputs.to(self.device)
        else:
            inputs = targets
            max_length = inputs.size(1)

        predicted_log_probs = []
        hidden_states = None
        # is_teacher_forcing = np.random.rand() < teacher_forcing_ratio
        is_teacher_forcing = False
        if is_teacher_forcing:
            step_outputs, hidden_states = self.forward_step(inputs, encoder_outputs, hidden_states)
            predicted_log_probs = step_outputs
        else:
            inputs = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                step_outputs, hidden_states = self.forward_step(inputs, encoder_outputs, hidden_states)
                predicted_log_probs.append(step_outputs)
                inputs = predicted_log_probs[-1].topk(1)[1]

        predicted_log_probs = torch.stack(predicted_log_probs, dim=1)
        return predicted_log_probs

    def forward_step(
            self,
            inputs: Tensor,
            encoder_outputs: Tensor,
            hidden_states: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        batch_size, output_length = inputs.size()[:2]

        embedding = self.embedding(inputs)
        if self.training:
            self.rnn.flatten_parameters()
        rnn, hidden_states = self.rnn(embedding, hidden_states)

        # test
        # encoder_outputs = encoder_outputs.view(-1)[:1280].view((1, -1, 640)).contiguous()
        attention = self.attention(rnn, encoder_outputs, encoder_outputs)
        outputs = torch.cat((rnn, attention), dim=2)
        step_outputs = self.fc(outputs.view(-1, self.hidden_size * 2)).log_softmax(dim=-1)
        step_outputs = step_outputs.view(batch_size, output_length, -1).squeeze(1)

        return step_outputs, hidden_states
