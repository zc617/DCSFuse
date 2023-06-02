import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import PIL
from PIL import Image
#from flowlib import read, read_weights_file
from skimage import io, transform
from PIL import Image
import numpy as np
import re
from uitils import *
import matplotlib.pyplot as plt


def toString(num):
    string = str(num)
    while (len(string) < 4):
        string = "0" + string
    return string


class ImageDataset(Dataset):

    def __init__(self, infrared_dataroot, visible_dataroot, image_size):
        """

        """
        self.infrared_dataroot = infrared_dataroot
        self.visible_dataroot = visible_dataroot
        self.image_size = image_size
        self.total_image = []
    
        for item in os.listdir(os.path.join(self.infrared_dataroot)):

            ir_img_dir = self.infrared_dataroot
            ir_image_list = os.listdir(os.path.join(self.infrared_dataroot))
            ir_image_list.sort(key=lambda x: str(re.split('\.|\_', x)[1]))
            vis_img_dir = self.visible_dataroot
            vis_image_list = os.listdir(os.path.join(self.visible_dataroot))
            vis_image_list.sort(key=lambda x: str(re.split('\.|\_', x)[1]))
            tmp_len = len(vis_image_list) - 1
        for i in range(tmp_len):
            ir_img = os.path.join(ir_img_dir, ir_image_list[i])
            vis_img = os.path.join(vis_img_dir, vis_image_list[i])
            tmp_image = (ir_img, vis_img)
            self.total_image.append(tmp_image)
        self.lens = len(self.total_image)
        self.transform = transforms.Compose([
            transforms.CenterCrop(self.image_size), #crop images
            transforms.ToTensor()])
           

    def __len__(self):
        return self.lens

    def __getitem__(self, i):
        """
        idx must be between 0 to len-1
        assuming flow[0] contains flow in x direction and flow[1] contains flow in y
        """
        ir_path1 = self.total_image[i][0]
        vis_path2 = self.total_image[i][1]
    
        ir_img1 = Image.open(ir_path1).convert('L')
        vis_img2 = Image.open(vis_path2).convert('L')

        ir_img1 = self.transform(ir_img1)
        vis_img2 = self.transform(vis_img2)
  
        return (ir_img1, vis_img2)


if __name__ == "__main__":
    ir_root="./TNO_Train/ir/"
    vi_root="./TNO_Train/vi/"
    image = ImageDataset(ir_root, vi_root,[128,128])
    print('data lens', len(image))
    dataloader = torch.utils.data.DataLoader(image, batch_size=1)
    for index, item in enumerate(dataloader):
        print(index)
        print(item[0].shape)
        print(item[1].shape)




