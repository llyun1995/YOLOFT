# 编程机构：长沙理工大学土木工程学院
# 编程人员：李凌云
# 编程时间：2024/09/8 0008 11:12
import torch
import torch.nn as nn


# autopad function as defined earlier
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# Conv class as defined earlier
class SiLU(nn.Module):
    # SiLU activation function
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Conv(nn.Module):
    # Standard convolution + BatchNorm + Activation
    default_act = SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act    = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


# Coordinate Attention implementation (CA)
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAttention, self).__init__()

        # c×1×W
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 自适应全局池化，输出的高度为1，宽度为可变
        # c×H×1
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 自适应全局池化，输出的宽度为1，高度为可变

        temp_c = max(8, in_channels // reduction)  # reduction 控制缩减比率，最小值为 8

        # 通道数降维
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        # 分别处理 H 和 W 维度的特征
        self.conv_h = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)  # c×H×1
        self.conv_w = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)  # c×1×W

    def forward(self, x):
        short = x
        n, c, H, W = x.shape

        # 处理 H 维度的特征
        x_h = self.pool_h(x)  # n×c×H×1
        # 处理 W 维度的特征，permute 调整维度顺序
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # n×c×1×W

        # 拼接 H 和 W 维度的特征
        x_cat = torch.cat([x_h, x_w], dim=2)  # n×c×(H+W)×1
        out = self.act1(self.bn1(self.conv1(x_cat)))

        # 将拼接后的特征再拆分成 H 和 W 部分
        x_h, x_w = torch.split(out, [H, W], dim=2)

        # 重新调整 W 维度特征的形状
        x_w = x_w.permute(0, 1, 3, 2)

        # 分别对 H 和 W 维度特征进行卷积并通过 Sigmoid 激活
        out_h = torch.sigmoid(self.conv_h(x_h))  # 高度方向的注意力
        out_w = torch.sigmoid(self.conv_w(x_w))  # 宽度方向的注意力

        # 最终输出乘以 H 和 W 的注意力权重
        return short * out_w * out_h


# Revised RFEM with CA integrated
class RFEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RFEM, self).__init__()

        # Branch 1: 1x1 conv -> 3x3 conv (dilation=1) -> CA
        self.branch1 = nn.Sequential(
            Conv(in_channels, out_channels, k=1),
            Conv(out_channels, out_channels, k=3, p=1, d=1),
            # CoordAttention(out_channels, out_channels)
        )

        # Branch 2: 1x1 conv -> 3x3 conv -> 3x3 conv (dilation=3) -> CA
        self.branch2 = nn.Sequential(
            Conv(in_channels, out_channels, k=1),
            Conv(out_channels, out_channels, k=3, p=1),
            Conv(out_channels, out_channels, k=3, p=3, d=3),
            # CoordAttention(out_channels, out_channels)
        )

        # Branch 3: 1x1 conv -> 3x3 conv -> 3x3 conv -> 3x3 conv (dilation=6) -> CA
        self.branch3 = nn.Sequential(
            Conv(in_channels, out_channels, k=1),
            Conv(out_channels, out_channels, k=3, p=1),
            Conv(out_channels, out_channels, k=3, p=1),
            Conv(out_channels, out_channels, k=3, p=6, d=6),
            # CoordAttention(out_channels, out_channels)
        )

        # 1x1 conv to adjust feature channels after concat
        self.concat_conv = Conv(out_channels * 3, out_channels, k=1)

        # CANet
        self.CANet = CoordAttention(out_channels, out_channels)

        # Shortcut connection (Branch 4)
        self.shortcut = Conv(in_channels, out_channels, k=1)

        # Output 1x1 conv (不含激活函数，因为后续会与shortcut相加再激活)
        self.output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

        # Activation function
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        # Branch outputs with CA applied
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)

        # Concatenate branch outputs
        concat_out = torch.cat([out1, out2, out3], dim=1)

        # Adjust channels after concat
        concat_out = self.concat_conv(concat_out)

        # 经过CANet调整注意力权重
        concat_out = self.CANet(concat_out)

        # Shortcut (Branch 4)
        shortcut = self.shortcut(x)

        # Add shortcut and concatenated output
        out = self.output_conv(concat_out) + shortcut

        # Apply SiLU activation
        out = self.silu(out)

        return out
