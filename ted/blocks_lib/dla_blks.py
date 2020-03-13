#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from os.path import join

import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
from ted.blocks_lib import *

BatchNorm = nn.BatchNorm2d

WEB_ROOT = 'DLA/pretrained_weights'


class BasicBlock(AbstractBaseBlock):
    def __init__(self, inplanes, planes, stride=1, dilation=1, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.conv1 = self.conv(inplanes, planes, kernel_size=3,
                        stride=stride, padding=dilation,
                        bias=False, dilation=dilation)
        self.bn1 = BatchNorm(planes)
#         self.relu = nn.ReLU(inplace=True)
        self.relu = self.act()
        self.conv2 = self.conv(planes, planes, kernel_size=3,
                        stride=1, padding=dilation,
                        bias=False, dilation=dilation)
        self.bn2 = BatchNorm(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)

        return out

class Bottleneck(AbstractBaseBlock):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = self.conv(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = self.conv(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = self.conv(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
#         self.relu = nn.ReLU(inplace=True)
        self.relu = self.act()
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

class BottleneckX(AbstractBaseBlock):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1, **kwargs):
        super(BottleneckX, self).__init__(**kwargs)
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = self.conv(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = self.conv(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = self.conv(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
#         self.relu = nn.ReLU(inplace=True)
        self.relu = self.act()
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(AbstractBaseBlock):
    def __init__(self, in_channels, out_channels, kernel_size, residual, **kwargs):
        super(Root, self).__init__(**kwargs)
        self.conv = self.conv(
            in_channels, out_channels, kernel_size,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = BatchNorm(out_channels)
#         self.relu = nn.ReLU(inplace=True)
        self.relu = self.act()
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(AbstractBaseBlock):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False, **kwargs):
        super(Tree, self).__init__(**kwargs)
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation, **kwargs)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation, **kwargs)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual, **kwargs)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual, **kwargs)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual, **kwargs)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                self.conv(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x