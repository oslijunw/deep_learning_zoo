import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import numpy as np
import torch.optim as optim
from model import AlexNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform_ops = {
    "train" : transforms.Compose(
        [   transforms.Resize(224),
            transforms.RandomCrop(224), # 随机裁剪
            transforms.RandomHorizontalFlip(), # 随机水平翻转
            transforms.ToTensor(), # 转化为张量，调整为C H W模式
            transforms.Normalize((.5, .5, .5), (.5, .5, .5)) # 三通道
        ],
    ),
    "val" : transforms.Compose(
        [
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ],
    ),
}

train_dataset = datasets.ImageFolder('./data/flowers', transform=transform_ops['train'])
train_dataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

epochs = 5

model = AlexNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    train_accurate = 0
    for step, batch in enumerate(train_dataLoader):
        batch_acc = 0 
        images, labels = batch
        optimizer.zero_grad()
        pred = model(images)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()
        pred_label = torch.max(pred, dim=1)[1]
        batch_acc += (pred_label == labels).numpy().sum()/len(images)
        epoch_loss += loss.item()
        print(f'epoch {epoch}, step {step} : loss {loss.item()}, batch accuracy {batch_acc}')
        
    print(f'epoch acculate loss: {epoch_loss}')
