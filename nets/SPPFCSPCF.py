# 编程机构：长沙理工大学土木工程学院
# 编程人员：李凌云
# 编程时间：2024/10/21 0021 14:33
import torch
import torch.nn as nn
from typing import Optional


def make_divisible(
        value: float,
        divisor: int,
        min_value: Optional[float] = None,
        round_down_protect: bool = True,
) -> int:
    """
    This function is copied from here
    "https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_layers.py"
    This is to ensure that all layers have channels that are divisible by 8.
    Args:
        value: A `float` of original value.
        divisor: An `int` of the divisor that need to be checked upon.
        min_value: A `float` of  minimum value threshold.
        round_down_protect: A `bool` indicating whether round down more than 10%
        will be allowed.
    Returns:
        The adjusted value in `int` that is divisible against divisor.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class SPPFCSPCF(nn.Module):
    # 本代码由YOLOAir目标检测交流群 心动 大佬贡献
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5, expand_ratio=4):
        super(SPPFCSPCF, self).__init__()
        bottleneck_channels = int(2 * c2 * e)  # Compressed channels
        expand_filters = make_divisible(c1 * expand_ratio, 8)

        # First inverted bottleneck
        self.cv1 = Conv(c1, expand_filters, 1, 1)   # Expand
        self.cv3 = Conv(expand_filters, expand_filters, 3, 1, g=expand_filters)   # Depthwise
        self.cv4 = Conv(expand_filters, bottleneck_channels, 1, 1, act=False)  # Compress

        # Max pooling for multi-scale feature extraction
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        # Second inverted bottleneck
        self.cv5 = Conv(4 * bottleneck_channels, bottleneck_channels, 1, 1)   # Adjust after concat
        self.cv6 = Conv(bottleneck_channels, bottleneck_channels, 3, 1, g=bottleneck_channels)  # Depthwise
        self.cv7 = Conv(2 * bottleneck_channels, c2, 1, 1, act=False)  # Compress final output

        # Second branch for skip connection
        self.cv2 = Conv(c1, bottleneck_channels, 1, 1)

    def forward(self, x):
        # First inverted bottleneck
        x1 = self.cv4(self.cv3(self.cv1(x)))

        # Multi-scale pooling
        x2 = self.m(x1)
        x3 = self.m(x2)
        x4 = self.m(x3)

        # Second inverted bottleneck with concatenation
        y1 = self.cv6(self.cv5(torch.cat((x1, x2, x3, x4), 1)))

        # Skip connection
        y2 = self.cv2(x)

        # Final output concatenation
        return self.cv7(torch.cat((y1, y2), dim=1))