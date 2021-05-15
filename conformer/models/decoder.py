from torch import nn, Tensor


class ConformerDecoder(nn.Module):
    """ Conformer decoder """

    decoder_dict = {
        'lstm': nn.LSTM
    }

    def __init__(self, decoder_name: str = 'lstm', input_size: int = 80, hidden_size: int = 4, num_layers: int = 4):
        super().__init__()
        assert decoder_name in self.decoder_dict.keys(), f"Not supported decoder name ({decoder_name})"

        if decoder_name == 'lstm':
            self.decoder = self.decoder[decoder_name](
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True
            )

    def forward(self, inputs: Tensor):
        return self.decoder(inputs)
