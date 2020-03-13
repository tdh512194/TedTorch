import torch
from torch import nn
from functools import partial
from ted.CoordCNN import CoordConvNet
from ted.blocks_lib.core import *
from ted.blocks_lib.dense_blks import *
from ted.blocks_lib.nonlocal_blks import *
from ted.blocks_lib.dla_blks import *
from ted.blocks_lib.classifier_blks import *
from ted.blocks_lib.acts import *

def choose_conv_block(block, conv=nn.Conv2d, **kwargs):
    return partial(block, conv=conv)