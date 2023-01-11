import torch 
import torch.nn as nn 
import torch.nn.functional as F

class Residual_Block(nn.Module):
    expansion = 4 #256/64
    def __init__(self,in_channel,out_channel,stride=1,downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,stride=1,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,stride=stride,padding=1,kernel_size=3,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel*self.expansion,stride=1,kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self,x):
        identity = x
        if self.downsample is not None:
            identity =self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity     # 残差连接
        out = self.relu(out)
        return out
    

class Resnet(nn.Module):
    def __init__(self,block,num_block,num_class=10):
        super().__init__()
        self.in_channel = 64
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=self.in_channel,stride=2,kernel_size=7,bias=False,padding=3) 
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self.make_layer(block=block,num_block=num_block[0],stride=1,out_channels=64)
        self.layer2 = self.make_layer(block=block,num_block=num_block[1],stride=2,out_channels=128)
        self.layer3 = self.make_layer(block=block,num_block=num_block[2],stride=2,out_channels=256)
        self.layer4 = self.make_layer(block=block,num_block=num_block[3],stride=2,out_channels=512)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion,num_class)


        for m in self.modules():    # 权重初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def make_layer(self,block,out_channels,num_block,stride=1):
        downsample = None
        if stride!=1 or self.in_channels != out_channels*block.expansion: #利用大小为1x1的卷积层扩展输入的维度后再进行相加,否则x与F(x)加不起来的
            downsample = nn.Sequential(nn.Conv2d(self.in_channels,out_channels*block.expansion,stride=stride,kernel_size=1,bias=False),
                                        nn.BatchNorm2d(out_channels*block.expansion)
                                        )

        layer = []
        layer.append(block(in_channels=self.in_channels, out_channels=out_channels, downsample=downsample, stride=stride))
        self.in_channels = out_channels * block.expansion
        for i in range(1,num_block):
            layer.append(block(in_channels=self.in_channels,out_channels=out_channels))

        return nn.Sequential(*layer)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet50():
    return Resnet(Residual_Block, [3, 4, 6,3])