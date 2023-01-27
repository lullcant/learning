import torch 
from torch import nn
from torch.nn import functional as F

class basic_block(nn.Module):
    def __init__(self,in_channel,out_channel) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        out = F.relu(x)
        return out

class Unetpp(nn.Module):
    def __init__(self,num_class,in_channel=3,deep_supervise=False) -> None:
        super().__init__()

        self.X_00 = basic_block(3,32)
        self.X_10 = basic_block(32,64)
        self.X_20 = basic_block(64,128)
        self.X_30 = basic_block(128,256)
        self.X_40 = basic_block(256,512)

        self.X_01 = basic_block(32+64,32)
        self.X_11 = basic_block(64+128,64)
        self.X_21 = basic_block(128+256,128)
        self.X_31 = basic_block(256+512,256)

        self.X_02 = basic_block(2*32+64,32)
        self.X_12 = basic_block(2*64+128,64)
        self.X_22 = basic_block(2*128+256,128)

        self.X_03 = basic_block(3*32+64,32)
        self.X_13 = basic_block(3*64+128,64)

        self.X_04 = basic_block(4*32+64,32)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.deepsupervision = deep_supervise
        
        if deep_supervise == True:
            self.out1 = nn.Conv2d(32,3,kernel_size=1)
            self.out2 = nn.Conv2d(32,3,kernel_size=1)
            self.out3 = nn.Conv2d(32,3,kernel_size=1)
            self.out4 = nn.Conv2d(32,3,kernel_size=1)
        else:
            self.out = nn.Conv2d(32,3,kernel_size=1)
    def forward(self,x):
        x0_0 = self.X_00(x)
        x1_0 = self.X_10(self.pool(x0_0))
        x0_1 = self.X_01(torch.concat([x0_0,self.up(x1_0)],dim=1))

        x2_0 = self.X_20(self.pool(x1_0))
        x1_1 = self.X_11(torch.concat([x1_0,self.up(x2_0)],dim=1))
        x0_2 = self.X_02(torch.concat([x0_0,x0_1,self.up(x1_1)],dim=1))

        x3_0 = self.X_30(self.pool(x2_0))
        x2_1 = self.X_21(torch.concat([x2_0,self.up(x3_0)],dim=1))
        x1_2 = self.X_12(torch.concat([x1_0,x1_1,self.up(x2_1)],dim=1))
        x0_3 = self.X_03(torch.concat([x0_0,x0_1,x0_2,self.up(x1_2)],dim=1))

        x4_0 = self.X_40(self.pool(x3_0))
        x3_1 = self.X_31(torch.concat([x3_0,self.up(x4_0)],dim=1))
        x2_2 = self.X_22(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.X_13(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.X_04(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))


        if self.deepsupervision:
            out1 = self.out1(x0_1)
            out2 = self.out2(x0_2)
            out3 = self.out3(x0_3)
            out4 = self.out4(x0_4)
            return [out1, out2, out3, out4]

        else:
            out = self.out(x0_4)
            return out


