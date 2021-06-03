from dataclasses import dataclass


@dataclass
class ConformerLargeConfig:
    encoder_dim: int = 512
    num_encoder_layers: int = 17
    num_attention_heads: int = 8
    conv_kernel_size: int = 31
    dropout_p: float = 0.1
