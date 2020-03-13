# TedTorch
Pytorch Deep Learning blocks and architectures built using abstract classes for ease of configuration

Abstract classes:
- `AbstractBaseBlock`
  - `AbstractBaseArchitecture`
- `AbstractBaseClassifier`
- `AbstractBaseSequential`

Configurable blocks:
- Convolution blocks: `nn.Conv2d` or `CoordConv`
- Activation function: `Mish`, `Swish`, `ESwish`, `Mila`
- Norm: `BatchNorm2d`, `InstanceNorm2d`

Supported Architectures:
- Densenet
- Deep Aggregation Layer

Ultilities:
- `Flatten` layer
- `Tedquential`: extension of `nn.Sequential` with configurable blocks and skip connections: `dense` and `residual`
