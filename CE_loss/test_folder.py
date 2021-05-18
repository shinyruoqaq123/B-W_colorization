import torch
from torch.autograd import Variable
from skimage.color import lab2rgb
from model import Color_model
from unet import UNet
#from data_loader import ValImageFolder
import numpy as np
from skimage.color import rgb2lab, rgb2gray
import torch.nn as nn 
from PIL import Image
import scipy.misc
from torchvision import datasets, transforms
from training_layers import decode
import torch.nn.functional as F
import os
import argparse
import cv2

scale_transform = transforms.Compose([
    # transforms.Scale(256),
    # transforms.RandomCrop(224),
])

def load_image(image_path,transform=None):
    image = Image.open(image_path)
    w,h=image.size
    if transform is not None:
        image = transform(image)
    if h<w:
        new_h=256
        rate=256/h
        new_w=int(w*rate)
    else:
        new_w=256
        rate=256/w
        new_h=int(float(h*rate))
    # print("h={},w={}".format(h,w))
    # print(("new_h={},new_w={}".format(new_h,new_w)))
    image=transforms.Resize((new_h//32*32,new_w//32*32))(image)
    if len(image.size)==2:
        image=image.convert("RGB")
    image=rgb2lab(image)[:,:,0]-50.
    image=torch.from_numpy(image).unsqueeze(0)
    
    return image

model_dict={
    'CIC':{'structure':Color_model,'down_rate':4},
    'UNet':{'structure':UNet,'down_rate':2}
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CIC', help='choose')
    parser.add_argument('--checkpoint_path', type=str, default='../model/models/', help='path for saving trained models')
    parser.add_argument('--test_folder_path', type=str, default='../model/models/', help='path for saving trained models')
    parser.add_argument('--output_folder_path', type=str, default='../model/models/', help='path for saving trained models')

    # Model parameters
    args = parser.parse_args()
    data_dir = args.test_folder_path
    dirs=os.listdir(data_dir)
    # color_model = nn.DataParallel(Color_model()).cuda().eval()
    color_model = nn.DataParallel(model_dict[args.model]['structure']()).cuda().eval()
    color_model.load_state_dict(torch.load(args.checkpoint_path))
     
    for file in dirs:
        image=load_image(data_dir+'/'+file)
        image=image.unsqueeze(0).float().cuda()
        # img_ab_313=color_model(image)
        img_ab_313 = color_model(image)
        out_max=np.argmax(img_ab_313[0].cpu().data.numpy(),axis=0)
        # print('out_max',set(out_max.flatten()))
        color_img=decode(image,img_ab_313,rate=model_dict[args.model]['down_rate'])
        #print(color_img)
        #break
        color_name = args.output_folder_path + file
        cv2.imwrite(color_name,color_img[:,:,::-1]*255)

if __name__ == '__main__':
    main()
