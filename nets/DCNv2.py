# 编程机构：长沙理工大学土木工程学院
# 编程人员：李凌云
# 编程时间：2024/09/2 0002 11:45
import torch
from torch import nn


class DConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        新增modulation 参数： 是DCNv2中引入的调制标量
        """
        super(DConv2d, self).__init__()
        self.kernel_size = kernel_size  # 卷积核的大小
        self.padding = padding  # 填充大小
        self.stride = stride  # 步幅大小
        self.zero_padding = nn.ZeroPad2d(padding)  # 使用零填充
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        # ---------------------------------------------------------------------------
        # p_conv：用于计算偏移量卷积层，输出通道是2N
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        # 权重初始化为0
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)
        # ---------------------------------------------------------------------------
        # 如果需要进行调制
        self.modulation = modulation
        if modulation:
            # m_conv：用于计算调制标量的卷积层，输出通道是N
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            # 权重初始化为0
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)  # 在指定网络层执行完backward()之后调用钩子函数hook

    # ---------------------------------------------------------------------------
    # 一个静态方法，用于在反向传播过程中对梯度进行调整。它将梯度缩放为原来的0.1倍。
    # 这是通过register_backward_hook方法注册到偏移量卷积层和调制卷积层上的。
    @staticmethod
    def _set_lr(module, grad_input, grad_output):  # 设置学习率的大小
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    # ---------------------------------------------------------------------------
    # 计算标准卷积核的相对坐标。p_n表示卷积核中每个点相对于卷积核中心的坐标偏移。
    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
                       torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                       torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # ---------------------------------------------------------------------------
    # 计算输入特征图每个卷积中心的基础坐标位置p_0。
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
                       torch.arange(1, h * self.stride + 1, self.stride),
                       torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    # ---------------------------------------------------------------------------
    # 计算卷积操作的实际坐标p，考虑偏移量offset、标准卷积核的相对坐标p_n 和基础坐标p_0，p是最终偏移后的采样点位置。
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)  # 偏移方向: (1, 2N, 1, 1)
        p_0 = self._get_p_0(h, w, N, dtype)  # 每个点的坐标：(1, 2N, h, w)
        p = p_0 + p_n + offset  # 最终有偏移的点的位置，表示将偏移方向、中心点坐标和偏移量相加，得到每个卷积
        return p  # (1,2N,h,w)

    # ---------------------------------------------------------------------------
    # 根据给定的采样位置q从输入特征图x中获取采样值，利用bilinear interpolation的思想获取邻近的像素值。
    # x_q：输入特征图上按照p坐标采样得到的特征值。
    # x_offset：经过双线性插值后的特征图，表示偏移坐标处的插值特征值，用于卷积计算。
    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    # ---------------------------------------------------------------------------
    # 将偏移后的特征图x_offset重新排列成标准卷积所需的格式，使其可以与标准卷积操作兼容。
    # 通过将x_offset从(b, c, h, w, N)重新排列为(b, c, h * ks, w * ks)，
    # 它将偏移后的采样点转换为一个标准的卷积输入格式，这样就可以直接应用标准卷积操作来处理这些采样特征。
    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        # 输入:x_offset是一个5D张量，大小为 (b, c, h, w, N)
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)], dim=-1)
        # 输出:x_offset被重新排列成一个4D张量，大小为(b, c, h * ks, w * ks)，其中ks是卷积核的大小(例如3x3卷积核的ks=3)。
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)
        return x_offset

    def forward(self, x):  # x: (b,c,h,w)
        # 首先通过p_conv计算偏移量offset：(b,2N,h,w)。学习到的偏移量，2N表示在x轴方向的偏移和在y轴方向的偏移
        offset = self.p_conv(x)
        # ---------------------------------------------------------------------------
        # 如果需要调制，则通过m_conv和torch.sigmoid函数计算调制标量m：(b,N,h,w)，学习到的N个调制标量，并将其应用到偏移特征图上。
        # 这样做可以进一步增强网络的表达能力，使其能够学习每个偏移位置的最优加权值。
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)
        # ---------------------------------------------------------------------------
        # 然后根据偏移量计算偏移后的采样点坐标p：(b, h, w, 2N)，其中2N表示每个位置有N个采样点，每个采样点包含x和y两个坐标。
        # p = p_0 + p_n + offset
        p = self._get_p(offset, dtype)
        p = p.contiguous().permute(0, 2, 3, 1)  # (b, h, w, 2N)
        # ---------------------------------------------------------------------------
        # 计算采样位置的四个角点，即计算采样点的四个整数位置索引(q_lt, q_rb, q_lb, q_rt)
        q_lt = p.detach().floor()  # 使用floor()函数取每个偏移点坐标的下界(左上角)，并使用detach()确保在反向传播时不会跟踪梯度。
        q_rb = q_lt + 1  # 上取整，通过对q_lt加1，得到右下角的索引q_rb。

        # 使用torch.clamp函数将计算出来的坐标裁剪到特征图的有效范围内，防止索引越界。x.size(2)和x.size(3)分别是输入特征图的高度和宽度。
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        # q_lb和q_rt分别表示左下角和右上角的整数位置索引，结合了q_lt和q_rb的部分坐标。
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        # 这样四个索引位置(q_lt, q_rb, q_lb, q_rt)对应于每个偏移点坐标的四个邻近整数点，用于双线性插值。
        # ---------------------------------------------------------------------------
        # 对最终的采样位置p的x和y坐标分别使用clamp函数，将它们限制在特征图的范围内。这样可以避免在计算双线性插值时越界。
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)
        # ---------------------------------------------------------------------------
        # 计算双线性插值的系数g_lt, g_rb, g_lb, g_rt，大小均为(b, h, w, N)
        # 计算这些系数时，使用了插值的公式，权重的计算是基于采样点与相邻网格点之间的相对距离。
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
        # ---------------------------------------------------------------------------
        # 获取采样点在原始图片中对应的真实特征值，大小均为(b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        # ---------------------------------------------------------------------------
        # 利用双线性插值，从输入特征图x中获取偏移后的特征x_offset，大小均为(b, c, h, w, N)
        # 插值步骤确保了偏移采样在网格间隙之间平滑过渡，从而提供更灵活、更精准的特征学习能力。
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt
        # ---------------------------------------------------------------------------
        # 如果需要调制，则计算调制标量m并应用到x_offset上。
        if self.modulation:  # m: (b,N,h,w)
            m = m.contiguous().permute(0, 2, 3, 1)  # (b,h,w,N)
            m = m.unsqueeze(dim=1)  # (b,1,h,w,N)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)  # (b,c,h,w,N)
            x_offset *= m  # 为偏移添加调制标量
        # ---------------------------------------------------------------------------
        # 重新排列x_offset并通过标准卷积conv生成最终输出out。
        '''
        偏置点含有九个方向的偏置，_reshape_x_offset()把每个点9个方向的偏置转化成3×3的形式，
        于是就可以用3×3 stride=3的卷积核进行Deformable Convolution，
        它等价于使用1×1的正常卷积核(包含了这个点9个方向的context)对原特征直接进行卷积。
        '''
        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out


# if __name__ == '__main__':
#     deformconv2d = DConv2d(64, 128, kernel_size=3, padding=1, stride=2, bias=None, modulation=False)
#     _input = torch.ones((1, 64, 5, 5))
#     result = deformconv2d(_input)
#     print(result.shape)
#     print(result)