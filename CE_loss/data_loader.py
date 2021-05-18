from torchvision import datasets, transforms
from skimage.color import rgb2lab, rgb2gray
from skimage import io
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

scale_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
    #transforms.ToTensor()
])


class TrainImageFolder(data.Dataset):
    def __init__(self, list_file_path,down_rate):
        list_file=open(list_file_path)
        self.file_list=list_file.readlines()
        self.down_rate=down_rate
    def __getitem__(self, index):
        #try:
            img=Image.open(self.file_list[index].strip())
            w,h=img.size
            if h < w:
                new_h = 256
                rate = 256 / h
                new_w = int(w * rate)
            else:
                new_w = 256
                rate = 256 / w
                new_h = int(float(h * rate))

            img = transforms.Resize((new_h // 32 * 32, new_w // 32 * 32))(img)
            img = transforms.RandomCrop(224)(img)
            img_original = transforms.RandomHorizontalFlip()(img)
            w, h = img_original.size
            img_resize=transforms.Resize((h//self.down_rate,w//self.down_rate))(img_original)
            img_original = np.asarray(img_original)
            img_lab = rgb2lab(img_resize)
            #img_lab = (img_lab + 128) / 255
            img_ab = img_lab[:, :, 1:3]
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1)))
            #print('img_ori',img_original.shape)
            #print('img_ab',img_ab.size())
            img_original = rgb2lab(img_original)[:,:,0]-50.
            img_original = torch.from_numpy(img_original)
            return img_original, img_ab
        #except:
        #    pass
    def __len__(self):
        return len(self.file_list)


class ValImageFolder(data.Dataset):
    def __init__(self,data_dir):
        self.file_list=os.listdir(data_dir)
        self.data_dir=data_dir

    def __getitem__(self, index):
        img=Image.open(self.data_dir+'/'+self.file_list[index])
        img_scale = scale_transform(img)
        img_scale = np.asarray(img_scale)
        img_scale = rgb2gray(img_scale)
        img_scale = torch.from_numpy(img_scale)
        return img_scale

    def __len__(self):
        return len(self.file_list)
