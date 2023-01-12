from torch.utils.data import Dataset
import torch
from PIL import Image 
import os
from torchvision import transforms
class Imageset(Dataset):
    def __init__(self,path,image_shape=(64,64)):
        super().__init__()
        self.path = path
        self.image_shape = image_shape
        self.name = sorted(os.listdir(self.path))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        img_name = self.name[index]
        img_path = os.path.join(self.path,img_name)
        img = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([transforms.CenterCrop(168),transforms.Resize(self.image_shape),transforms.ToTensor()])

        return transform(img)
