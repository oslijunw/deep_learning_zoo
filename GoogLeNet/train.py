import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from model import GoogLeNet


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

train_dataset = datasets.ImageFolder(root='../data/PetImages/train', transform=transform)
train_dataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
i2c = train_dataset.class_to_idx

test_dataset = datasets.ImageFolder(root='../data/PetImages/test', transform=transform)
test_dataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = GoogLeNet(num_class=2)
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    net.train()
    for index, (images, labels) in enumerate(train_dataLoader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        pred, pred_aux2, pred_aux1 = net(images)
        loss_aux2 = loss_fn(pred_aux2.to(device), labels)
        loss_aux1 = loss_fn(pred_aux1.to(device), labels)
        loss = loss_fn(pred.to(device), labels)
        loss = loss + .3*loss_aux1 + .3*loss_aux2

        loss.backward()
        optimizer.step()
        print(f'iter {index} loss:', loss.item())

    net.eval()
    accuracy = 0
    for images, labels in test_dataLoader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            pred = net(images)

            pred = torch.max(pred, dim=1)[1]
            pred.to(device)
            accuracy = int(accuracy + (labels == pred).clone().detach().sum())
            # print(labels == pred)
            print('batch accuracy num: ', int(accuracy))
    accuracy = accuracy/len(test_dataset)
    print(f'epoch{epoch} accuracy : {accuracy}')
    torch.save(net.state_dict(), f'./models/{epoch}.pth')




