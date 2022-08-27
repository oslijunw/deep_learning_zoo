from hashlib import new
from re import T
import re
from stringprep import in_table_c11
from tkinter.messagebox import NO
from unittest import removeResult
from unittest.util import _MIN_COMMON_LEN
import torch
import torch.nn as nn


class ConvBNReLU6(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size-1) // 2
        super().__init__(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                # groups=1 普通卷积，groups=in_channel dw卷积
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio) -> None:
        super().__init__()
        # 中间dw卷积的输入特征矩阵大小
        hidden_channel = in_channel * expand_ratio


        self.short_cut_available = stride == 1 and out_channel == in_channel

        layers = []
        if expand_ratio != 1:
            # 如果expand_ratio == 1 不改变channel也不改变矩阵大小，作用不大，所以忽视
            # 1x1 pointwise conv
            layers.append(
                ConvBNReLU6(in_channel=in_channel, out_channel=hidden_channel, kernel_size=1)
            )
        layers.expand([
            # 3x3 depthwise conv
            ConvBNReLU6(
                in_channel=hidden_channel,
                out_channel=hidden_channel,
                kernel_size=3,
                groups=hidden_channel,
                stride=stride
            ),
            # 1x1 pointwise conv(linear,由于y=x,所以BN后就不需要操作)
            nn.Conv2d(
                in_channels=hidden_channel,
                out_channels=out_channel,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channel)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.short_cut_available:
            return x + self.conv(x)
        else:
            return self.conv(x)


def _make_divisible(pre_channel_num, divisor=8, min_ch=None):
    """
    大致是为了将chanel调整为8的倍数
    """
    if min_ch is None:
        min_ch = divisor
    # // 自动向下取整，max第二个参数，相当于寻找距离pre_channel_num最近的8的倍数的channel
    new_ch = max(min_ch, int(pre_channel_num + divisor/2)//divisor*divisor)
    if new_ch < .9 * pre_channel_num:
        # 确保减少的不超过10%
        new_ch += divisor
    return new_ch


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8) -> None:
        super().__init__()
        
        block = InvertedResidual
        # 第一个block开始的输入channel
        in_channel = _make_divisible(32*alpha, round_nearest)
        # 全部block卷积之后的channel
        last_channel = _make_divisible(1280*alpha, round_nearest)

        inverted_residual_config = [
            # t c n s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        features.append(
            # 第一个卷积层，非倒残差结构，out_channel是为了得到一个8倍数的channel，写死32也行，只是类似于统一处理32*alpha
            ConvBNReLU6(in_channel=3, out_channel=in_channel, stride=2)
        )
        # build inverted residual blocks
        for t, c, n, s in inverted_residual_config:
            out_channel = _make_divisible(c*alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                inverted_residual_block = block(in_channel=in_channel, out_channel=out_channel, stride=stride)
                features.append(inverted_residual_block)
                in_channel = out_channel

        # 1x1 conv after bottleneck
        # last_channel 硬编码成1280也行
        features.append(ConvBNReLU6(in_channel=in_channel, out_channel=last_channel, kernel_size=1))
        
        self.features = nn.Sequential(*features)
        
        # 每一个channel都转化为1x1的样子
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(.2),
            nn.Linear(last_channel, num_classes)
        )
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, .01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


    def forward(self, x):
        x = self.features(x)
        x = self.avg_pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x