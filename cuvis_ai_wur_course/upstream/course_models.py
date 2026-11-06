"""Course model classes from ``end2end_spectral_vs_spatial_models.ipynb``.

Source: notebook ``end2end_spectral_vs_spatial_models.ipynb``, commit ``c65ba36`` of
``LizaGenke/Chemometric-approaches-to-hyperspectral-processing``
(https://github.com/LizaGenke/Chemometric-approaches-to-hyperspectral-processing).
Licence pending the authors' written confirmation.

Kept as the course wrote it: the standardisation modules, the spectral regressor
wrapper, the MLP, and the generic 1D/2D/3D ResNet used for the 1D ResNet spectral
model. Only imports, docstrings, and formatting were touched so the course's
``best_weights.pt`` state dicts for ``mlp`` and ``resnet1d`` load with
``strict=True`` against these classes. See the module docstring of
``cuvis_ai_wur_course.node.models`` for the list of changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class InputStandardizer(nn.Module):
    """(x - mean) / std with scalar mean and std stored as buffers (saved in the checkpoint)."""

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        super().__init__()
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(float(std)))

    def forward(self, x):
        return (x - self.mean) / self.std


class OutputDestandardizer(nn.Module):
    """x * std + mean: maps the z-scored fat value the backbone predicts back to percent fat."""

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        super().__init__()
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(float(std)))

    def forward(self, x):
        return x * self.std + self.mean


class SpectrumRegressor(nn.Module):
    """Standardise the spectrum, apply the backbone (MLP or 1D ResNet), de-standardise the output.
    add_channel_dim=True turns (B, L) into (B, 1, L), the layout a 1D convolution expects."""

    def __init__(
        self, backbone, add_channel_dim=False, mean=0.0, std=1.0, out_mean=0.0, out_std=1.0
    ):
        super().__init__()
        self.standardizer = InputStandardizer(mean, std)
        self.backbone = backbone
        self.add_channel_dim = add_channel_dim
        self.destandardizer = OutputDestandardizer(out_mean, out_std)

    def forward(self, x):
        x = self.standardizer(x)
        if self.add_channel_dim:
            x = x.unsqueeze(1)
        x = self.backbone(x)
        return self.destandardizer(x)


def get_norm_layer_1d(normalization, num_features):
    """Normalisation over the feature dimension of a (B, F) tensor: 'bn', 'ln' or None (identity)."""
    if normalization == "bn":
        return nn.BatchNorm1d(num_features, momentum=0.01)
    if normalization == "ln":
        return nn.LayerNorm(num_features)
    if normalization is None:
        return nn.Identity()
    raise ValueError("normalization must be None, 'bn' or 'ln'")


class MLP(nn.Module):
    """A stack of (Linear -> normalisation -> ReLU -> Dropout) blocks followed by a linear head."""

    def __init__(
        self, input_dim=300, hidden_dims=(256, 128), output_dim=1, normalization=None, dropout=0.0
    ):
        super().__init__()
        # A bias in the linear layer is only needed when no normalisation layer follows
        bias = normalization is None
        layers = []
        in_features = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_features, hidden, bias=bias))
            layers.append(get_norm_layer_1d(normalization, hidden))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(in_features, output_dim)

    def forward(self, x):
        return self.head(self.features(x))


def conv_nd(dim):
    return {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dim]


def conv3x3(dim, in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 (or length-3) convolution with padding 1, so the size only changes through the stride."""
    return conv_nd(dim)(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )


def conv1x1(dim, in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution: mixes channels without looking at neighbours. Used to match shortcut shapes."""
    return conv_nd(dim)(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class Permute(nn.Module):
    """Permute the dimensions of a tensor (used to apply LayerNorm over the channel axis)."""

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


def get_norm_layer(normalization, channels, dim):
    """Batch norm ('bn'), channel-wise layer norm ('ln') or nothing (None) for a dim-D feature map."""
    if normalization == "bn":
        return {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}[dim](
            channels, momentum=0.01
        )
    if normalization == "ln":
        p_in = {1: (0, 2, 1), 2: (0, 2, 3, 1), 3: (0, 2, 3, 4, 1)}[dim]
        p_out = {1: (0, 2, 1), 2: (0, 3, 1, 2), 3: (0, 4, 1, 2, 3)}[dim]
        return nn.Sequential(Permute(p_in), nn.LayerNorm(channels), Permute(p_out))
    if normalization is None:
        return nn.Identity()
    raise ValueError("normalization must be None, 'bn' or 'ln'")


class BasicBlock(nn.Module):
    """conv3x3 -> norm -> ReLU -> conv3x3 -> norm, plus the shortcut, then ReLU."""

    expansion = 1

    def __init__(
        self,
        dim,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        normalization="ln",
    ):
        super().__init__()
        bias = normalization is None
        self.conv1 = conv3x3(dim, inplanes, planes, stride, bias=bias)
        self.bn1 = get_norm_layer(normalization, planes, dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(dim, planes, planes, bias=bias)
        self.bn2 = get_norm_layer(normalization, planes, dim)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))  # F(x), first half
        out = self.bn2(self.conv2(out))  # F(x), second half
        if self.downsample is not None:  # match the shape of the shortcut
            identity = self.downsample(x)
        out += identity  # the residual connection: F(x) + x
        return self.relu(out)


class ResNet(nn.Module):
    """Generic ResNet supporting 1D, 2D or 3D convolutions, following the torchvision-style layout."""

    def __init__(self, input_channels, dim, num_classes, block, layers, normalization="ln"):
        super().__init__()
        self.dim = dim
        self.normalization = normalization
        bias = normalization is None
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        # Stem. In 1D the spectrum always arrives as a single channel (B, 1, L)
        self.conv1 = conv_nd(dim)(
            input_channels if dim != 1 else 1,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=bias,
        )
        self.bn1 = get_norm_layer(normalization, self.inplanes, dim)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = {1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}[dim](
            kernel_size=3, stride=2, padding=1
        )
        # Four stages, each halving the spatial (or spectral) size and doubling the channels
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # Global average pooling: one number per channel, whatever the input size
        self.avgpool = {
            1: nn.AdaptiveAvgPool1d(1),
            2: nn.AdaptiveAvgPool2d((1, 1)),
            3: nn.AdaptiveAvgPool3d((1, 1, 1)),
        }[dim]
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Kaiming (He) initialisation, designed for ReLU networks
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(
                    self.dim,
                    self.inplanes,
                    planes * block.expansion,
                    stride,
                    bias=self.normalization is None,
                ),
                get_norm_layer(self.normalization, planes * block.expansion, self.dim),
            )
        layers = [
            block(
                self.dim,
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                self.dilation,
                self.normalization,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.dim,
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    normalization=self.normalization,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


def resnet18(input_channels, dim, num_classes, **kwargs):
    """ResNet-18: four stages with two BasicBlocks each."""
    return ResNet(input_channels, dim, num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
