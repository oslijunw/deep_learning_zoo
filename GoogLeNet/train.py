
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from model import GoogLeNet


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg))
    plt.show()
    
train_dataset = datasets.ImageFolder(root='./data/PetImages', transform=transform)
train_dataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
net = GoogLeNet(num_class=2)
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    net.train()
    for index, (images, labels) in enumerate(train_dataLoader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        preds, preds_aux2, preds_aux1 = net(images)
        loss_aux2 = loss_fn(preds_aux2.to(device), labels)
        loss_aux1 = loss_fn(preds_aux1.to(device), labels)
        loss = loss_fn(preds.to(device), labels)
        loss = loss + .3*loss_aux1 + .3*loss_aux2

        loss.backward()
        optimizer.step()

        print(f'iter {index} loss:', loss.item())


