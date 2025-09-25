# 编程机构：长沙理工大学土木工程学院
# 编程人员：李凌云
# 编程时间：2024/09/8 0008 16:23
"""本文给大家带来的改进机制是由YOLO-Face提出能够改善物体遮挡检测的注意力机制SEAM，SEAM(Spatially Enhanced AttentionModule)注意力义网络模块旨在补偿被遞挡面部的响应损失，
通过增强未遮挡面部的响应来实现这一目标，其希望通过学习遮挡面和未遮挡面之间的关系来改善遮挡情况下的损失从而达到改善物体遮挡检测的效果"""
# 本文的改善遮挡的注意力机制我是建议用在Neck的输出部分来进行使用。
import torch
import torch.nn as nn

__all__ = ['SEAM', 'MultiSEAM']


class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class SEAM(nn.Module):
    def __init__(self, c1, n=1, reduction=16):
        super(SEAM, self).__init__()
        c2 = c1
        self.DCovN = nn.Sequential(
            # nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, groups=c1),
            # nn.GELU(),
            # nn.BatchNorm2d(c2),
            *[nn.Sequential(
                # 残差块
                Residual(nn.Sequential(
                    nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=1, padding=1, groups=c2),
                    nn.GELU(),
                    nn.BatchNorm2d(c2)
                )),
                # 1x1卷积层用于跨通道的信息整合
                nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=0, groups=1),
                nn.GELU(),
                nn.BatchNorm2d(c2)
            ) for i in range(n)]
        )
        # 自适应平均池化层(AdaptiveAvgPool2d)将每个通道的特征图压缩为一个全局特征向量。
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        # 全连接层(fc)实现了通道注意力机制，它使用了一个 Squeeze-and-Excitation(SE)结构，通过Linear层进行通道压缩与恢复，
        # 最后使用Sigmoid激活函数来生成每个通道的注意力权重。
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid()
        )

        self._initialize_weights()
        # self.initialize_layer(self.avg_pool)
        self.initialize_layer(self.fc)

    def forward(self, x):
        b, c, _, _ = x.size()
        # 输入x首先经过DCovN处理，生成空间增强的特征图，然后通过自适应平均池化和fc层计算出每个通道的注意力权重。
        y = self.DCovN(x)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # 最后，通过exp函数放大注意力权重，并将其与输入x相乘，从而增强重要的特征通道。
        y = torch.exp(y)
        return x * y.expand_as(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def initialize_layer(self, layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)


def DcovN(c1, c2, depth, kernel_size=3, patch_size=3):
    dcovn = nn.Sequential(
        nn.Conv2d(c1, c2, kernel_size=patch_size, stride=patch_size),
        nn.SiLU(),
        nn.BatchNorm2d(c2),
        *[nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=kernel_size, stride=1, padding=1, groups=c2),
                nn.SiLU(),
                nn.BatchNorm2d(c2)
            )),
            nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=0, groups=1),
            nn.SiLU(),
            nn.BatchNorm2d(c2)
        ) for i in range(depth)]
    )
    return dcovn


class MultiSEAM(nn.Module):
    def __init__(self, c1, depth=1, kernel_size=3, patch_size=[3, 5, 7], reduction=16):
        super(MultiSEAM, self).__init__()
        c2 = c1
        self.DCovN0 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[0])
        self.DCovN1 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[1])
        self.DCovN2 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[2])
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y0 = self.DCovN0(x)
        y1 = self.DCovN1(x)
        y2 = self.DCovN2(x)
        y0 = self.avg_pool(y0).view(b, c)
        y1 = self.avg_pool(y1).view(b, c)
        y2 = self.avg_pool(y2).view(b, c)
        y4 = self.avg_pool(x).view(b, c)
        y = (y0 + y1 + y2 + y4) / 4
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)
        return x * y.expand_as(x)