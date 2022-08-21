import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs
        )
        self.relu = nn.ReLU(True)
    
    def forward(self, x):
        return self.relu(self.conv(x))


class Inception(nn.Module):
    """
    Inception结构,以融合不同尺度的特征信息
    比如拿着不同的卷积核进行卷积,并将其进行融合操作
    但是需要确保不同的特征信息最终的结果是可融合的(其实就是大小一致)
    """
    def __init__(
        self, 
        in_channels,
        channel_1x1,
        channel_3x3_reduce,
        channel_3x3,
        channel_5x5_reduce,
        channel_5x5,
        channel_pool_proj
    ) -> None:
        super().__init__()
        self.branch_1x1 = BasicConv2d(
            in_channels=in_channels,
            out_channels=channel_1x1,
            kernel_size=1
        )

        self.branch_3x3 = nn.Sequential(
            BasicConv2d(
                in_channels=in_channels,
                out_channels=channel_3x3_reduce,
                kernel_size=1
            ),
            # 保证输入输出大小一致
            # out_size = (in_size - 3 + 2 * 1) / 1 + 1
            # 相当于 kernel_size-1=2*padding
            BasicConv2d(
                in_channels=channel_3x3_reduce,
                out_channels=channel_3x3,
                kernel_size=3,
                padding=1
            ) 
        )

        self.branch_5x5 = nn.Sequential(
            BasicConv2d(
                in_channels=in_channels,
                out_channels=channel_5x5_reduce,
                kernel_size=1
            ),
            BasicConv2d(
                in_channels=channel_5x5_reduce,
                out_channels=channel_5x5,
                kernel_size=5,
                padding=2
            ) 
        )

        self.branch_pooling = nn.Sequential(
            # 保证输入和输出大小一致，从而在1*1卷积下就能保持不变
            nn.MaxPool2d(
                kernel_size=3,
                stride=1,
                padding=1
            ),
            BasicConv2d(
                in_channels=in_channels,
                out_channels=channel_pool_proj,
                kernel_size=1
            )
        )
    
    def forward(self, x):
        branch_1x1 = self.branch_1x1(x)
        branch_3x3 = self.branch_3x3(x)
        branch_5x5 = self.branch_5x5(x)
        branch_pooling = self.branch_pooling(x)

        outputs = [branch_1x1, branch_3x3, branch_5x5, branch_pooling]
        # [batch, channel, height, weight]
        return torch.cat(outputs, dim=1)


class AuxiliaryClassifier(nn.Module):
    """
    辅助分类器
    """
    def __init__(self, in_channels, num_class) -> None:
        super().__init__()
        self.avg_pooling = nn.AvgPool2d(kernel_size=5, stride=3)
        # 池化不改变channel数
        self.conv = BasicConv2d(
            in_channels=in_channels,
            out_channels=128,
            kernel_size=1
        )
        self.fc1 = nn.Linear(128*4*4, 1024)
        self.fc2 = nn.Linear(1024, num_class)

    def forward(self, x):
        # aux1 [N 512 14 14] aux2 [N 528 14 14]
        x = self.avg_pooling(x)
        # aux1 [N 512 4 4] aux2 [N 528 4 4]
        x = self.conv(x)
        # [N 128 4 4]

        x = torch.flatten(x, start_dim=1)
        # self.training 由net.train()和net.eval()控制
        # 这个是作用在展开层上的
        x = F.dropout(x, .5, training=self.training)

        # [N 2048]
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, .5, training=self.training)

        # [N 1024]
        # 最后一层一般由softmax当激活函数，这边不需要操作
        x = self.fc2(x)

        # [N num_class]
        return x


class GoogLeNet(nn.Module):
    """
    1x1的卷积核用作降维以及映射操作
    一共有三个分类器，增加两个辅助分类器辅助训练
    丢弃全连接层，使用平均池化层，以降低模型的参数
    """
    def __init__(self, num_class:int=1000, aux_logits:bool = True, init_weight:bool=True) -> None:
        super().__init__()
        self.aux_logits = aux_logits

        # 经过conv1卷积后，size变为一半 224->112
        self.conv1 = BasicConv2d(
            in_channels=3, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3
        )
        # ceil_mode 结果向上取整，否则向上取整
        self.pooling_max_1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # nn.LocalResponseNorm在这里被简化了
        self.conv2_3 = nn.Sequential(
            BasicConv2d( # 3x3 reduce
                in_channels=64,
                out_channels=64,
                kernel_size=1
            ),
            BasicConv2d(
                in_channels=64,
                out_channels=192,
                kernel_size=3,
                padding=1
            )
        )
        self.pooling_max_2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # 每个Inception层的in_channels可以看上一层
        # 如果是普通层，就没什么好顾虑的
        # 如果是inception其实就是不同特征信息channel总和
        self.inception3a = Inception(
            in_channels=192,
            channel_1x1=64,
            channel_3x3_reduce=96,
            channel_3x3=128,
            channel_5x5_reduce=16,
            channel_5x5=32,
            channel_pool_proj=32
        )
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.pooling_max_3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # 128 + 192 + 96 + 64 = 480，参数图里各层out channel就是这么计算的 
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.pooling_max_4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        
        if self.aux_logits:
            # 辅助分类器
            self.aux1 = AuxiliaryClassifier(512, num_class=num_class)
            self.aux2 = AuxiliaryClassifier(528, num_class=num_class)

        # 不论输入是什么，都能得到1x1的矩阵，相当于展平操作
        # 这边的效果,实际上是将每个channel都变成一个平均值
        self.pooling_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(.4)
        self.fc = nn.Linear(1024, num_class)

        if init_weight:
            self._init_weights()
        
    def forward(self, x):
        # N 3 224 224
        x = self.conv1(x)
        # N 64 112 112
        x = self.pooling_max_1(x)
        # N 64 56 56 -> N 64 56 56
        x = self.conv2_3(x)
        # N 64 192 56 56
        x = self.pooling_max_2(x)

        # N 192 28 28
        x = self.inception3a(x)
        # N 256 28 28
        x = self.inception3b(x)
        # N 480 28 28
        x = self.pooling_max_3(x)

        # N 480 14 14
        x = self.inception4a(x)
        # N 512 14 14
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
        x = self.inception4b(x)
        # N 512 14 14
        x = self.inception4c(x)
        # N 512 14 14
        x = self.inception4d(x)
        # N 528 14 14
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
        x = self.inception4e(x)
        # N 832 14 14
        x = self.pooling_max_4(x)
        
        # N 832 7 7
        x = self.inception5a(x)
        # N 832 7 7
        x = self.inception5b(x)
        
        # N 1024 7 7
        x = self.pooling_avg(x)
        x = torch.flatten(x, 1)

        # N 1024 1 1
        x = self.dropout(x)
        x = self.fc(x)

        # N num_class
        if self.training and self.aux_logits:
            return x, aux2, aux1
        
        return x

    def _init_weights(self):
        """
        尽管GoogLeNet比较复杂,但是本质还是由卷积和全连接层构成的
        池化操作不涉及参数,所以只要初始化卷积层参数和全连接层参数即可
        """
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(
                    layer.weight,
                    mode="fan_out"
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, .01)
                nn.init.constant_(layer.bias, 0)

    