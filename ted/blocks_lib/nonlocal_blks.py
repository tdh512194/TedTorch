import torch
from torch import nn
from ted.blocks_lib import *

class NonlocalBlock(AbstractBaseBlock):
    """
    similar to Self-Attention mechanism
    Non-local Neural Networks
    https://arxiv.org/pdf/1711.07971.pdf
    """
    def __init__(self, in_features, hidden_features=1024, **kwargs):
        super(NonlocalBlock, self).__init__(**kwargs)
        self.base = self._make_conv_layer(in_features, hidden_features, kernel_size=1)
        self.tail = self._make_conv_layer(hidden_features // 2, hidden_features, kernel_size=1)
        self.query = self._make_conv_layer(hidden_features, hidden_features // 2, kernel_size=1)
        self.key = self._make_conv_layer(hidden_features, hidden_features // 2, kernel_size=1)
        self.value = self._make_conv_layer(hidden_features, hidden_features // 2, kernel_size=1)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        identity = self.base(x)
        
        query = self.query(identity)
        key   = self.key(identity)
        value = self.value(identity)

        score = torch.matmul(query, key)
        score = self.softmax(score)
        
        z = torch.matmul(score, value)
        z = self.tail(z)
        z += identity
        return z