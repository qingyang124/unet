import torch
import torchvision.transforms.functional
from torch import nn


# 两次卷积操作
class DoubleConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # 第一次3*3、无padding的卷积
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.act1 = nn.ReLU()
        # 第2次3*3、无padding的卷积，第一次卷积输出通道数成为第二次卷积输入通道数
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # 应用2次卷积和线性修正
        x = self.first(x)
        x = self.act1(x)
        x = self.second(x)
        return self.act2(x)


# 下采样
class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        # 2*2最大值池化操作
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        return self.pool(x)


# 上采样
class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # 转置卷积
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.up(x)


# 特征图的裁剪和连接
class CropAndConcat(nn.Module):
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        # 裁剪压缩路径上的特征图 ？x.shape[2], x.shape[3]？
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        # 连接压缩路径特征图和扩展路径特征图
        x = torch.cat([x, contracting_x], dim=1)


# UNet 模型搭建
class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # 压缩路径上的每两次卷积操作
        self.down_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
                                        [(in_channels, 64), (64, 128), (128, 256), (256, 512)]])

        # 压缩路径下采样过程，最大值池化操作
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])

        # U型最下面那层卷积
        self.middle_conv = DoubleConvolution(512, 1024)

        # 上采样过程，转置卷积操作
        self.up_sample = nn.ModuleList([UpSample(i, o) for i, o in
                                        [(1024, 512), (512, 256), (256, 128), (128, 64)]])

        # 上采样过程，两次卷积操作
        self.up_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
                                      [(1024, 512), (512, 256), (256, 128), (128, 64)]])

        # 上采样过程，裁剪连接特征图操作
        self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])

        # 最后一次卷积该操作
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # 定义一个列表哦，存储压缩路径上的特征图，方便与后面扩展路径上特征图拼接
        pass_through = []
        # 压缩路径
        for i in range(len(self.down_conv)):
            # 每层2次卷积
            x = self.down_conv[i](x)
            # 存储每次卷积结果
            pass_through.append(x)
            x = self.down_sample[i](x)

        # U型最下面那层3*3卷积
        x = self.middle_conv(x)

        # 扩展路径
        for i in range(len(self.up_conv)):
            # 上采样 转置卷积
            x = self.up_sample[i](i)
            # 连接压缩路径上的特征图
            x = self.concat[i](x, pass_through.pop())
            # 两次3*3卷积
            x = self.up_conv[i](x)

        # 最后一次1*1卷积
        x = self.final_conv(x)

        # 返回结果
        return x