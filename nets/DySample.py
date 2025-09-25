# 编程机构：长沙理工大学土木工程学院
# 编程人员：李凌云
# 编程时间：2024/09/8 0008 23:46
"""本文介绍了DySample，一种轻量级且高效的动态上采样器，它优于传统内核上采样器，减少参数和计算量。
   DySample在语义分割和目标检测等任务中表现出色，通过点采样方法实现，降低了GPU内存需求。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------
# 这两个函数用于初始化模型中的卷积层参数：
# normal_init: 初始化卷积层的权重为正态分布，均值为mean，标准差为std，并初始化偏置为bias。
# constant_init: 将卷积层的权重和偏置初始化为常数，权重为val，偏置为bias。
def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


# -------------------------------------------------------------
class Dy_Sample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale  # scale:上采样的倍率，默认为2，表示将输入特征图尺寸放大2倍。
        self.style = style  # style:上采样的样式，有两种选项 lp 和 pl。
        self.groups = groups  # groups:通道分组数，默认为4。
        assert style in ['lp', 'pl']

        # lp:代表Learnable Position，直接学习位置偏移量。
        if style == 'lp':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        # pl:代表Position Learned from Pixel Shuffle，基于Pixel Shuffle(像素移位)操作先进行上采样，然后再学习偏移量。
        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        # 使用1x1卷积生成位移偏移量(offset)，输出通道数为2*groups*scale**2。
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)

        # 是否引入额外的偏移调整机制，如果为True，会额外学习一个scope卷积层。
        if dyscope:
            # 可选的附加偏移范围调整机制(dyscope=True时启)用，也使用1x1卷积生成同样的位移偏移量。
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    # 位置初始化(_init_pos): 初始位置偏移量，是基于网格位置的静态偏移值。
    """该函数生成了一个初始偏移量init_pos，它是一个基于scale比例的固定网格，作用是在采样时提供基准位置。
       这有助于防止完全依赖网络学习位移值，提供一个合理的初始位置。"""
    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    # 核心采样函数(sample): sample函数通过学习的offset对输入特征图进行动态采样。
    def sample(self, x, offset):
        B, _, H, W = offset.shape
        # offset包含特征图每个位置的水平和垂直方向的位移。
        offset = offset.view(B, 2, -1, H, W)
        # 函数生成网格坐标coords，并将它们与offset相加，生成新的采样坐标。
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        # 最后，使用F.grid_sample在这些新的坐标上对输入特征图进行双线性插值，生成上采样后的特征图。
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    # 在lp模式下，偏移值通过self.offset卷积层生成，并通过self.scope(如果有)调整。然后使用sample函数根据偏移值对特征图进行采样。
    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    # 在pl模式下，先使用Pixel Shuffle将特征图上采样，然后计算偏移量，并执行动态采样。
    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


# if __name__ == '__main__':
#     x = torch.rand(2, 64, 4, 7)
#     dys = Dy_Sample(64)
#     print(dys(x).shape)