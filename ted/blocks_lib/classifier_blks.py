import torch
from torch import nn
from torch.nn import functional as F
from ted.blocks_lib import *


class ConcatPoolClassifier(AbstractBaseClassifier):
    """
    Concatenation of Avg and Max Pooling
    Commonly used for varied size input
    """
    def __init__(self, in_features, num_classes, **kwargs):
        super(ConcatPoolClassifier, self).__init__(**kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.flatten = Flatten()
        in_feat_1 = in_features*(1 if self.is_flat else 2)
        in_feat_2 = in_features//(2 if self.is_flat else 1)
        self.classifier = nn.Sequential(
            self._make_classifier(in_feat_1, in_feat_2, use_act=True, p=0.25),
            self._make_classifier(in_feat_2, num_classes, use_act=False, p=0.5),
        )
        self.classifier.apply(self.init_weights)

    def forward(self, x):
        if not self.is_flat:
            _avg = self.avgpool(x)
            _max = self.maxpool(x)
            x = torch.cat([_avg, _max], dim=1)
            x = self.flatten(x)
        x = self.classifier(x)
        return x

class NonlocalPoolClassifier(AbstractBaseClassifier):
    """
    Combination of NonlocalBlock and ConcatPoolClassifier
    Self-Attention -> ConcatPool
    """
    def __init__(self, in_features, num_classes, **kwargs):
        super(NonlocalPoolClassifier, self).__init__(**kwargs)
        self.nolocal = NonlocalBlock(in_features=in_features, hidden_features=in_features, **kwargs)
        self.poolclassifier = ConcatPoolClassifier(in_features, num_classes, **kwargs)
        self.nolocal.apply(self.init_weights)
        self.poolclassifier.apply(self.init_weights)

    def forward(self, x):
        x = self.nolocal(x)
        x = self.poolclassifier(x)
        return x