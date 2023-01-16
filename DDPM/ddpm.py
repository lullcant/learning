import numpy as np
import torch
from torch import nn
from typing import Tuple, Optional
import torch.nn.functional as F
import torch.utils.data

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

class Denoise_Diffusion:
    def __init__(self,eps_model:nn.Module,n_steps:int,device:torch.device) -> None:
        super().__init__()
        #the eps_model will always be Unet
        self.eps_model = eps_model
        #Now, in order to make the forward process probability becomes an istropic Gaussian Distribution, we create increasing B_t
        self.beta = torch.linspace(0.001,0.02,n_steps).to(device) 
        #set alpha to be 1-beta for convenience of calculation
        self.alpha = 1-self.beta
        # alphabar = Pi alpha
        self.alphabar = torch.cumprod(self.alpha,dim=0)
        #T=n_step
        self.n_steps = n_steps
        #define the variance of gaussian
        self.sigma2 = self.beta

        ##Now we can get the conditional distribution given q(xt|x0)
    def q_xt_x0 (self,x0,t):
        mean = gather(self.alphabar,t)**0.5*x0
        var = gather(1-self.alphabar,t)
        return mean,var
    ##sampling using reparameterization technique

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps == None:
            eps = torch.randn_like(x0) #sample z from standard normal
        mean,var = self.q_xt_x0(x0,t) #calculate mean and var of q(xt|x0)
        res = mean + (var**0.5)*eps  #y = mean + std*eps
        return res
    
    ##Now we get the conditional distr
    def p_sample(self,xt,t):
        eps_theta = self.eps_model(xt, t)
        alpha_bar = gather(self.alphabar,t)
        alpha = gather(self.alpha,t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** (0.5) ###  beta/sqrt(1-alpha_bar) 
        mean = 1/alpha**0.5*(xt-eps_coef*eps_theta)
        var = gather(self.sigma2,t)
        ##perform the reparameterization agagin
        eps = torch.randn(xt.shape,device=xt.device) #no need xt device
        return eps*var**0.5+mean
    
    def loss(self,x0) :
        batch_size = x0.shape[0]
        if noise is None:
            noise = torch.randn_like(x0)
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        xt = self.q_sample(x0, t, eps=noise)
        eps_theta = self.eps_model(xt, t)
        return F.mse_loss(noise,eps_theta)
