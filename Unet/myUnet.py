import numpy as np
import torch 
import torch.nn as nn 
from torch.nn import functional as F
class basic_block(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1): 
        super().__init__()
        self.layer = nn.Sequential(
        nn.Conv2d(in_channels=in_channel,out_channels=out_channel,stride=1,kernel_size=3,padding=1),#symmetric padding, increase the ability
        nn.BatchNorm2d(out_channel),
        nn.Dropout(0.3),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(in_channels=out_channel,out_channels=out_channel,stride=1,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_channel),
        nn.Dropout(0.3),
        nn.LeakyReLU(inplace=True)
        )
    def forward(self,x):
        return self.layer(x)

#down sample in the paper
class Downsample(nn.Module):
    def __init__(self,in_channel,out_channel,stride=2):
        super().__init__()
        self.downsample = nn.MaxPool2d(kernel_size=2)
        self.conv = basic_block(in_channel,out_channel)
    def forward(self,x):
        x = self.downsample(x)
        out = self.conv(x)
        return out

##down sample using 3x3 convollution block(保存特征)
# class Downsample(nn.Module):
#     def __init__(self,in_channel,out_channel,stride=2):
#         super().__init__()
#         self.layer = nn.Sequential(nn.Conv2d(in_channels=in_channel,out_channels=out_channel,stride=2,padding=1,kernel_size=3,padding_mode='reflect',bias=False),
#         nn.BatchNorm2d(out_channel),
#         nn.LeakyReLU(inplace=True)
#         )
    
#     def forward(self,x):
#         return self.layer(x)

#up conv is not simply convolution, we shall use 插值法
class Up(nn.Module):
    def __init__(self,in_channel,out_channel,stride=2) :
        super().__init__()
        self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv = basic_block(in_channel,out_channel)
    def forward(self,x,feature_map):
        x = self.up(x)
        diffY = feature_map.size()[2] - x.size()[2]
        diffX = feature_map.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x =torch.cat([x,feature_map],dim=1)
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self,num_class):
        super().__init__()
        self.c1 = basic_block(3,64)
        self.d1 = Downsample(64,128)
        # self.c2 = basic_block(64,128)
        self.d2 = Downsample(128,256)
        # self.c3 = basic_block(128,256)
        self.d3 = Downsample(256,512)
        # self.c4 = basic_block(256,512)
        self.d4 = Downsample(512,512)
        # self.c5 = basic_block(512,1024)
        self.u1 = Up(1024,512//2)
        # self.c6 = basic_block(1024,512)
        self.u2 = Up(512,256//2)
        # self.c7 = basic_block(512,256)
        self.u3 = Up(256,128//2)
        # self.c8 = basic_block(256,128)
        self.u4 = Up(128,64)
        # self.c9 = basic_block(128,64)
        self.out = nn.Conv2d(64,num_class,kernel_size=1)
        # self.th = nn.Sigmoid()


    def forward(self,x):
        R1 = self.c1(x) ##first feature map
        R2 = self.d1(R1)
        R3 = self.d2(R2)
        R4 = self.d3(R3)
        R5 = self.d4(R4)
        #print(R5.shape,R4.shape)
        # up sampling
        R6 = self.u1(R5,R4)
        R7 = self.u2(R6,R3)
        R8 = self.u3(R7,R2)
        R9 = self.u4(R8,R1)
        out = self.out(R9)
       # out = self.th(out)
        return out
    
if __name__ == '__main__':
    x = torch.randn([2,3,256,256])
    model = Unet(3)
    print(model(x).shape)
