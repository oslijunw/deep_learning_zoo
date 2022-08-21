import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    18 layers or 34 layers ResNet's residual block
    expansion: 残差结果所使用卷积核的倍数变化
               1乘上去没有作用
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None) -> None:
        # downsample 下采样，残差处理，是否用1x1卷积降维 
        super().__init__()
        # 有BN层，不需要去增加bias，加不加效果一样
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(
            num_features=out_channels
        )
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 下采样方法
        self.downsample = downsample

    def forward(self, x):
        shortcut = x

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # 两个矩阵各个维度一样大小，直接相加拼接 
        out += shortcut

        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    用于50层、101层、152层的ResNet
    expansion: 残差结果所使用卷积核的倍数变化
               在这三种层次的残差模块
               最后一次卷积的卷积核是前面两个卷积的卷积核的四倍
               所以设置为4
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            stride=1, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, stride=stride, bias=False, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels*self.expansion,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.relu = nn.ReLU()

        self.downsample = downsample

    def forward(self, x):
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += shortcut

        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet
    """
    def __init__(self, block:BasicBlock or Bottleneck, blocks_num:list, num_class:int=1000, include_top:bool=True) -> None:
        """
        block : residual module -> BasicBlock or Bottleneck
        blocks_num : block num
        num_class: default 1000 due to Image Net class num
        include_top : 便于基于ResNet搭建复杂网络 
        """
        super().__init__()
        self.include_top = include_top
        self.residual_block_in_channels = 64

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=self.residual_block_in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn = nn.BatchNorm2d(self.residual_block_in_channels)
        self.relu = nn.ReLU(True)
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.residual_block1 = self._make_layer(
            block=block, 
            residual_block_out_channel=64, 
            block_num=blocks_num[0]
        )
        self.residual_block2 = self._make_layer(
            block=block, 
            residual_block_out_channel=128, 
            block_num=blocks_num[1],
            stride=2
        )
        self.residual_block3 = self._make_layer(
            block=block, 
            residual_block_out_channel=256, 
            block_num=blocks_num[2],
            stride=2
        )
        self.residual_block4 = self._make_layer(
            block=block, 
            residual_block_out_channel=512, 
            block_num=blocks_num[3],
            stride=2
        )
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512*block.expansion, num_class)
        
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out')

    def _make_layer(
        self, 
        block:BasicBlock or Bottleneck,
        residual_block_out_channel,
        block_num,
        stride=1):
        """
        创建残差层
        """
        downsample = None
        if stride != 1 or self.residual_block_in_channels != residual_block_out_channel*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.residual_block_in_channels,
                    out_channels=residual_block_out_channel*block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(residual_block_out_channel*block.expansion)
            )
        
        layers = []
        layers.append(
            block(
                self.residual_block_in_channels, 
                residual_block_out_channel, 
                downsample=downsample, 
                stride=stride
            )
        )
        self.residual_block_in_channels = residual_block_out_channel * block.expansion

        for _ in range(1, block_num):
            layers.append(
                block(
                    self.residual_block_in_channels, 
                    residual_block_out_channel,
                )
            )
        
        return nn.Sequential(*layers)
  
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pooling(x)

        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        x = self.residual_block4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        
        return x


def resnet34(num_class=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_class=num_class, include_top=include_top)

def resnet101(num_class=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_class=num_class, include_top=include_top)

