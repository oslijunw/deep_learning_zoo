import torch.nn as nn
import torch

class AlexNet(nn.Module):
    def __init__(self, init_weight:bool=False, num_classes:int=1000) -> None:
        super().__init__()
        # 提取图片特征
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2), 
            # nn.ZeroPad2d((1, 2, 1, 2)), 精准填充
            nn.ReLU(inplace=True), # inplace是以时间换空间
            nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.denses = nn.Sequential(
            nn.Dropout(p=.5),

            nn.Linear(4608, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

        if init_weight:
            self._init_weights()

    def forward(self, x):
        x = self.backbone(x)
        # [batch, channel, height, weight]->start_dim=1 
        # view函数亦可
        x = torch.flatten(x, start_dim=1)
        x = self.denses(x)
        return x

    def _init_weights(self):
        for m in self.modules(): 
            # Returns an iterator over all modules in the network.
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, .01)
                nn.init.constant_(m.bias, 0)


