#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from os.path import join

import torch
from torch import nn
import torch.utils.model_zoo as model_zoo

from ted.archs_lib.dla_arch import dataset as dataset
from ted.blocks_lib import *


BatchNorm = nn.BatchNorm2d

WEB_ROOT = 'http://dl.yf.io/dla/models'

def get_model_url(data, name):
    return join(WEB_ROOT,'{}-{}.pth'.format(name, data.model_hash[name]))

class DLA(AbstractBaseArchitecture):
    """
    Deep Layer Aggregation
    https://arxiv.org/abs/1707.06484
    """
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, return_levels=False,
                 pool_size=7, linear_root=False, **kwargs):
        super(DLA, self).__init__(**kwargs)
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            self.conv(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            BatchNorm(channels[0]),
            self.act())
#             nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,root_residual=residual_root, **kwargs)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root, **kwargs)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root, **kwargs)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root, **kwargs)
        
        if not self.return_features:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.maxpool = nn.AdaptiveMaxPool2d(1)
            self.fc_avgmax = self.conv(channels[-1] * 2, num_classes, kernel_size=1,
                                stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                self.conv(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(planes),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample, **kwargs))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, **kwargs))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                self.conv(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                BatchNorm(planes),
                self.act()])
#                 nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        if self.return_levels:
            return y
        elif self.return_features:
            return x
        else:
            _avg = self.avgpool(x)
            _max = self.maxpool(x)
            x = torch.cat([_avg, _max], dim=1)
            x = self.fc_avgmax(x)
            x = x.view(x.size(0), -1)
            return x

    def load_pretrained_model(self, data_name, name):
        assert data_name in dataset.__dict__, \
            'No pretrained model for {}'.format(data_name)
        data = dataset.__dict__[data_name]
        fc = self.fc_avgmax
        if self.num_classes != data.classes:
            self.fc_avgmax = self.conv(
                self.channels[-1], data.classes,
                kernel_size=1, stride=1, padding=0, bias=True)
        try:
            model_url = get_model_url(data, name)
        except KeyError:
            raise ValueError(
                '{} trained on {} does not exist.'.format(data.name, name))
#         self.load_state_dict(model_zoo.load_url(model_url))
        self.load_state_dict(torch.load(model_url), strict=False)
        self.fc_avgmax = fc


def dla34(pretrained=None, **kwargs):  # DLA-34
    block = choose_conv_block(BasicBlock, **kwarg)
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=block, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla34')
    return model


def dla46_c(pretrained=None, **kwargs):  # DLA-46-C
    Bottleneck.expansion = 2
    block = choose_conv_block(Bottleneck, **kwarg)
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 64, 128, 256],
                block=block, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla46_c')
    return model


def dla46x_c(pretrained=None, **kwargs):  # DLA-X-46-C
    BottleneckX.expansion = 2
    block = choose_conv_block(BottleneckX, **kwarg)
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 64, 128, 256],
                block=block, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla46x_c')
    return model


def dla60x_c(pretrained=None, **kwargs):  # DLA-X-60-C
    BottleneckX.expansion = 2
    block = choose_conv_block(BottleneckX, **kwarg)
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 64, 64, 128, 256],
                block=block, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla60x_c')
    return model


def dla60(pretrained=None, **kwargs):  # DLA-60
    Bottleneck.expansion = 2
    block = choose_conv_block(Bottleneck, **kwarg)
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=block, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla60')
    return model


def dla60x(pretrained=None, **kwargs):  # DLA-X-60
    BottleneckX.expansion = 2
    block = choose_conv_block(BottleneckX, **kwarg)
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=block, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla60x')
    return model


def dla102(pretrained=None, **kwargs):  # DLA-102
    Bottleneck.expansion = 2
    block = choose_conv_block(Bottleneck, **kwargs)
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=block, residual_root=True, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla102')
    return model


def dla102x(pretrained=None, **kwargs):  # DLA-X-102
    BottleneckX.expansion = 2
    block = choose_conv_block(BottleneckX, **kwargs)
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=block, residual_root=True, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla102x')
    return model


def dla102x2(pretrained=None, **kwargs):  # DLA-X-102 64
    BottleneckX.cardinality = 64
    block = choose_conv_block(BottleneckX, **kwargs)
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=block, residual_root=True, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla102x2')
    return model


def dla169(pretrained=None, **kwargs):  # DLA-169
    Bottleneck.expansion = 2
    block = choose_conv_block(Bottleneck, **kwarg)
    model = DLA([1, 1, 2, 3, 5, 1], [16, 32, 128, 256, 512, 1024],
                block=block, residual_root=True, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla169')
    return model