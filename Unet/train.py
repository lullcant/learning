import torch 
from torch import nn
import torch.optim
from torch.utils.data import DataLoader,random_split
from torch.nn import functional as F
from mydata import *
import argparse
from myUnet import *
from tqdm import tqdm
from datametric import *
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_dir = './data/imgs/'
mask_dir = '/data/masks/model.pth'
weight_path = '/params'
def train(model=Unet(3),
        device=device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,####Normalization of loss
        momentum: float = 0.999,#### use to reduce the flipping of loss
        gradient_clipping: float = 1.0):### clipping the gradient during the back propagation to prevent gradient exploding
        try:
            dataset = CarvanaDataset(img_dir,mask_dir,scale=img_scale)
        except(AssertionError, RuntimeError, IndexError):
            dataset = Basic_Dataset(img_dir,mask_dir,scale=img_scale)
        
        #train test split
        n_val = int(val_percent*(len(dataset)))
        n_train = len(dataset)-n_val
        train_set,val_set = random_split(dataset,[n_train, n_val],generator=torch.Generator().manual_seed(0))
        
        #Dataloader creation
        loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
        train_data_loader = DataLoader(dataset=train_set,shuffle=True,**loader_args)
        validation_data_loader = DataLoader(dataset=val_set,shuffle=False,**loader_args)

        #optimizer
        optimizer = torch.optim.Adam(model.parameter(),lr=learning_rate,weight_decay=weight_decay, momentum=momentum, foreach=True)
        #loss
        criterion = nn.CrossEntropyLoss() if model.num_class else nn.BCEWithLogitsLoss()

        for i in range(1,epochs+1):
            model.train()
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                for batch in enumerate(train_data_loader):#batch is a dict
                    image,masks = batch['image'],batch['mask'] #材料都要送进device
                    image = image.to(device)
                    masks = masks.to(device)
                    pred = model(image)
                    if model.num_class==1:
                        loss = criterion(masks,pred.squeeze(1))
                        loss += dice_loss(F.sigmoid(pred.squeeze(1)),masks)
                    else:
                        loss = criterion(masks,pred)
                        loss += dice_loss(F.softmax(masks),F.one_hot(pred,model.num_class).permute(0,3,1,2).float,muticlass=True)######
                    print("training loss ",loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if i%5==0:
                        torch.save(model.state_dict(),weight_path)
if __name__=='__main__':
