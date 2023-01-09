import numpy as np
import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import logging
from functools import partial
from multiprocessing import Pool

def read_image(file):
    extra = os.path.splitext(file)[1]
    if extra == '.npy':
        img = Image.fromarray(np.load(file))
    elif extra == '.pth' or extra == '.pt':
        img = Image.fromarray(torch.load(file).numpy())
    else:
        img = Image.open(file)
    return img

def unique_mask(ids,mask_path,mask_suffix):
    mask_file = list(mask_path.glob(ids + mask_suffix + '.*'))[0]
    mask = np.asarray(read_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask,axis=0)
    else:
        raise ValueError(f'the maskdim should only be 2 or 3, but get {mask.ndim}')
def preprocess(mask_values,img,scale,is_mask):
    w,h = img.size
    newW, newH = img.size[0]*scale,img.size[1]*scale
    assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
    img_processed = img.resize((newW,newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
    img_numpy = np.asanyarray(img_processed)
    if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i ### all(-1）啥意思。

            return mask

    else:
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))

        if (img > 1).any(): ##img>1.any是true意味着至少有大于一的，即没有归一化
            img = img / 255.0

        return img
##first make the dataset
class Basic_Dataset(Dataset):
    def __init__(self,img_path,mask_path,scale=1.0,mask_suffix=''):
        super().__init__()
        self.img_path = os.path(img_path)
        self.mask_path = os.path(mask_path)
        assert 0 <scale <=1
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [os.path.splitext(file)[0] for file in os.listdir(img_path) if os.path.exists(os.path.join(img_path,file)) and not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        #f = partial(unique_mask,self.ids,self.mask_path,self.mask_suffix)
        with Pool() as p:
            unique = list(tqdm(p.imap(partial(unique_mask,self.ids,self.mask_path,self.mask_suffix),self.ids),total=len(self.ids)))
            self.mask_values = list(sorted(np.unique(np.concatenate(unique),axis=0).tolist()))##the concatenate function change n dimensional matrix to a row vector.
            logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)      
    
    def __getitem__(self, index):

        name = self.ids[index]
        img_file = list(self.img_path.glob(name  + '.*'))
        mask_file = list(self.mask_path.glob(name + self.mask_suffix + '.*'))
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        img = read_image(img_file[0])
        mask = read_image(mask_file[0])
        assert img.size == mask.size,f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        img = preprocess(self.mask_values,img,self.scale,False)
        mask = preprocess(self.mask_values,img,self.scale,True)


        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),#这里不如torch.Tenosr(img),as_tensor是共享内存的，img动它也动，但是写了img.copy就没啥意义
            'mask': torch.as_tensor(mask.copy()).long().contiguous() #contiguous 是对view函数方便，这个玩意重新开了数据位置一样且底层数组展开方式一样的连续tensor。
        }

class CarvanaDataset(Basic_Dataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')