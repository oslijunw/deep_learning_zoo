# 迁移学习的训练模式
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets


from model import MobileNetV2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

epochs = 10
batch_size = 64

data_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
     "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
}

train_dataset = datasets.ImageFolder(root='../data/PetImages/train', transform=data_transform['train'])

i2c = train_dataset.class_to_idx
with open('i2c.json', mode='w') as file:
    file.write(str(i2c))

test_dataset = datasets.ImageFolder(root='../data/PetImages/test', transform=data_transform['val'])


# num_worker = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
# print('Using {} dataloader workers every process'.format(num_worker))

train_dataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

net = MobileNetV2(num_classes=2)

model_weight_path = "./mobilenet_v2.pth"
assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
pre_weights = torch.load(model_weight_path, map_location=device)

pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

# freeze features weights
for param in net.features.parameters():
    param.requires_grad = False

net.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

best_acc = 0.0
save_path = './MobileNetV2.pth'
train_steps = len(train_dataLoader)
for epoch in range(epochs):
    # train
    net.train()
    running_loss = 0.0
    for data in train_dataLoader:
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_func(logits, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        print("train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss))

    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in test_dataLoader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            print("valid epoch[{}/{}]".format(epoch + 1, epochs))
    val_accurate = acc / len(test_dataLoader)
    print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' % (epoch + 1, running_loss / train_steps, val_accurate))

    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(net.state_dict(), save_path)

print('Finished Training')
