import torch
import torch.nn as nn


cfgs = {
    'vgg11':[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13':[64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

def make_features(cfg: list):
    """backbone"""
    layers = []
    # RGB
    in_channels = 3
    for l in cfg:
        if l == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=l,
                kernel_size=3,
                padding=1,
                stride=1
            ))
            layers.append(nn.ReLU(inplace=True))
            in_channels = l
    # unpackage
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features, class_num=1000, init_weight=False) -> None:
        super().__init__()
        self.backbone = features
        self.classifier = nn.Sequential(
            nn.Dropout(.5),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),

            nn.Dropout(.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),

            nn.Linear(4096, class_num)
        )
        if init_weight:
            ...
        
    def forward(self, x):
        x = self.backbone(x)

        x = torch.flatten(x, start_dim=1)

        x = self.classifier(x)
    
        return x


def vgg(model_name='vgg11', **kwargs):
    model = VGG(make_features(cfg=cfgs[model_name]), **kwargs)
    return model
