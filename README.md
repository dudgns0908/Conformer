# Conformer
PyTorch Implementation: [Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)

## **Working now...**


## Introduction
This is an ASR model called **Conformer** made by Google. <br />
This paper introduces only encoder models. However, I implemented both encoder and decoder model using PyTorch.  
Encoder was implemented as conformer according to the paper, and decoder was implemented as 'Something'. 
(Decoder has not been decided which model to use)

### Conformer Architecture
![Conformer Encoder Architecture](docs/images/encoder_block.png) <br />
description

### Feed Forward Module
![Feed Forward Module](docs/images/feed_forward_module.png) <br />
description

### Multi-Head Self Attention Module
![Multi-Head Self Attention Module](docs/images/multi_head_self_attention_module.png) <br />
description

### Convolution Module
![Convolution Module](docs/images/convolution_module.png) <br />
description


## Installation
### Usage
<h3>Train</h3>

```python
from conformer.trainer import Trainer
Trainer().fit(...)
```


## Reference
### Paper 
- **[Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)**

### Github
- **[sooftware/KoSpeech](https://github.com/sooftware/KoSpeech)**