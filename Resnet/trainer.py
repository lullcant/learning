import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import dataloader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
import resnetrf
from tqdm import tqdm
epoch = 1
parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('-b', '--batch_size', default=128, type=int,metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-l','--learning_rate', default=0.0001, type=float,metavar='LR', help='initial learning rate')
args = parser.parse_args()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) #golden value

train_data = dataloader.DataLoader(
    datasets.CIFAR10(root='../data', train=True, transform=transforms.Compose([
       # transforms.Resize(32, 32),      # 重新设置图片大小
        transforms.ToTensor(),      # 将图片转化为tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])         # 进行归一化
    ]), download=False), shuffle=True, batch_size=args.batch_size
)

# 导入测试集数据
train_test = dataloader.DataLoader(
    datasets.CIFAR10(root='../data', train=False, transform=transforms.Compose([
        #transforms.Resize(32, 32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=False), shuffle=True, batch_size=args.batch_size
)

#model = resnet.resnet50()
model = resnetrf.resnet50()
##define loss and optimizer
criteon = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

for i in range(epoch):

    model.train()
    for batch_idx,(image,label) in tqdm(enumerate(train_data),total=len(train_data)):
        image = image
        output = model(image)
        loss = criteon(output,label)
        #print('logits:', output[0])
        #print('label:', label[0])
        print('training loss',loss)
        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx == len(train_data) - 1:
            print(epoch, 'loss:', loss.item())

    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_num = 0
        for x, label in train_test:
            x = x
            label = label

            logits = model(x)

            pred = logits.argmax(dim=1)

            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)

        acc = total_correct / total_num
        print(epoch, 'test acc:', acc)

