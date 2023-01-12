import torch 
from torch import nn
import numpy as np

class VAE(nn.Module):
    def __init__(self,in_channel,hidden=[16,32,64,128,256],latent_dim=128):
        super().__init__()
        #encoder
        modules = []
        in_channel = 3
        imgsize = 64
        for current_hidden in hidden:
            modules.append(nn.Sequential(nn.Conv2d(in_channel,current_hidden,stride=2,kernel_size=3,padding=1),
            nn.BatchNorm2d(current_hidden),
            nn.ReLU(inplace=True)
            ))
            in_channel = current_hidden
            imgsize//=2
        self.encoder = nn.Sequential(*modules)
        self.meanlinear = nn.Linear(imgsize**2*in_channel,latent_dim)#全连接，inchannel个矩阵，每个矩阵imgsize*imgsize
        self.varlinear = nn.Linear(imgsize**2*in_channel,latent_dim)
        self.latent_dim = latent_dim
        #decoder
        modules2=[]
        self.projection = nn.Linear(latent_dim,imgsize**2*in_channel)
        self.decoder_input_chw = (in_channel, imgsize, imgsize)
        for i in range(len(hidden) - 1, 0, -1):#从len（hidden）-1开始到1结束。
            modules2.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden[i],
                                        hidden[i - 1],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        output_padding=1),
                    nn.BatchNorm2d(hidden[i - 1]), nn.ReLU()))
        modules2.append(
            nn.Sequential(
            nn.ConvTranspose2d(hidden[0],
                                        hidden[0],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        output_padding=1),
            nn.BatchNorm2d(hidden[0]), nn.ReLU(),
            nn.Conv2d(hidden[0], 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)))
        self.decoder = nn.Sequential(*modules2)

    def forward(self,x):
        z = self.encoder(x)
        mean = self.meanlinear(z)
        logvar = self.varlinear(z)
        std = torch.exp(logvar/2)
        eps = torch.rand_like(logvar)
        p = eps*std + mean
        x = self.projection(p)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))####
        x = self.decoder(x)
        
        return x,mean,logvar
