import math
import torch
from torch import nn
from functools import partial
from ted.CoordCNN import CoordConvNet

class AbstractBaseBlock(nn.Module):
    """
    base CNN blocks with configurable 
    conv, norm and activation layers
    """
    def __init__(self, conv=nn.Conv2d, norm=nn.BatchNorm2d, act=nn.ReLU, p=0.25, **kwargs):
        super(AbstractBaseBlock, self).__init__()
        self.conv = conv
        self.norm = norm
        self.p = p
        self.act = act

    def _make_conv_layer(self, in_features, hidden_features, kernel_size, 
                         use_norm=True, use_act=True, use_drop=False):
        layers = nn.ModuleList()
        layers.append(self.conv(in_features, hidden_features, kernel_size=kernel_size))
        layers.append(self.norm(hidden_features)) if use_norm else None
        layers.append(nn.Dropout2d(p=self.p)) if use_drop else None
        layers.append(self.act()) if use_act else None
        conv_layer = nn.Sequential(*layers)
        conv_layer.apply(self.init_weights)
        return conv_layer

    @staticmethod
    def init_weights(module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)


class AbstractBaseArchitecture(AbstractBaseBlock):
    """
    children class of AbstractBaseBlock
    `return_features`: return unflattened/unpooled CNN features map
    """
    def __init__(self, conv=nn.Conv2d, norm=nn.BatchNorm2d, act=nn.ReLU, p=0.25, return_features=False, **kwargs):
        super(AbstractBaseArchitecture, self).__init__(conv=conv, norm=norm, act=act, p=p, **kwargs)
        self.return_features = return_features


class AbstractBaseClassifier(nn.Module):
    """
    base class for classifiers
    use to set activations and dropout rates
    `is_flat` determined if need flattened
    """
    def __init__(self, p=0.25, act=nn.ReLU, is_flat=False, **kwargs):
        super(AbstractBaseClassifier, self).__init__()
        self.p = p
        self.act = act
        self.is_flat = is_flat

    def _make_classifier(self, in_features, out_features, 
                         use_norm=True, use_drop=True, 
                         use_act=False, p=None):
        """
        apply a (Batchnorm > dropout > fc > activation) combo by default
        """
        layers = nn.ModuleList()
        layers.append(nn.BatchNorm1d(in_features)) if use_norm else None
        p = self.p if p is None else p
        layers.append(nn.Dropout(p)) if use_drop else None
        layers.append(nn.Linear(in_features, out_features))
        layers.append(self.act()) if use_act else None

        classifier = nn.Sequential(*layers)
        classifier.apply(self.init_weights)
        return classifier
    @staticmethod
    def init_weights(module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

class AbstractBaseSequential(nn.Sequential):
    """
    Extension of `nn.Sequential` with configurable
    `conv`, `norm`, `act`, `dropout` and
    `merge` that allows creating skip connections [`residual`, `dense`]
    """
    def __init__(self, layers, conv=nn.Conv2d, norm=nn.BatchNorm2d, act=nn.ReLU, p=0.25, merge=None, **kwargs):
        super(AbstractBaseSequential, self).__init__(*layers)
        self.conv_, self.norm_, self.act_, self.p, self.merge = conv, norm, act, p, merge
    def forward(self, x):
        self.identity = x if self.merge else None
        x = super(AbstractBaseSequential, self).forward(x)
        if self.merge == 'residual':
            x += self.identity
        elif self.merge == 'dense':
            x = torch.cat([x, self.identity], dim=1)
        return x

class Tedquential(AbstractBaseSequential):
    def __init__(self, layers, merge=None):
        super(Tedquential, self).__init__(layers=layers, merge=merge)

class Flatten(nn.Module):
    """
    flatten layer before fc layers
    `keep_batch` is for making model layer, keeping the batch dimmension
    """
    def __init__(self, keep_batch=True): 
        super(Flatten, self).__init__(); self.keep_batch = keep_batch
    def forward(self, x): 
        return x.view(x.size(0), -1) if self.keep_batch else x.view(-1)
