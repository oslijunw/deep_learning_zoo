import torch
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision
# install dataset，aviod certificate verify failed
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))]
)
train_dataset = torchvision.datasets.CIFAR10(root='data/', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset)