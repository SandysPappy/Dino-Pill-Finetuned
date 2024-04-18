import argparse
import glob
import os
import pickle
import sys
from utils import utils
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dino import utils
from src.constants import COLORS
from src.plotting.vis_tools import visualize_attention
from dataset_loaders import get_epill_dataloader
import cv2
import random
import colorsys
import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch.nn as nn

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, image_size=224):
        '''
        flip_and_color_jitter = transforms.Compose([
            #transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        '''
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            #flip_and_color_jitter,
            #utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            #flip_and_color_jitter,
            #utils.GaussianBlur(0.1),
            #utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(int(image_size//2), scale=local_crops_scale, interpolation=Image.BICUBIC),
            #flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        to_pil = transforms.ToPILImage()
        image = to_pil(image)
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return transforms.ToTensor()(image), crops

def simple_imshow_tensor(im):
    # Assuming `img` is your image tensor with shape [3, 64, 64]
    im = im.permute(1, 2, 0)  # Change from CxHxW to HxWxC
    im = (im-im.min())/(im.max()-im.min())
    # If your image values are in [0, 255], you can normalize them to [0, 1]
    # im = im / 255.0
    plt.imshow(im)
    plt.axis('off')  # Hide axes for better visualization
    plt.show()

if __name__ == '__main__':

    # Call the above function to seed everything
    seed_everything()
    local_crops_number = 9
    batch_size = 32

    crop_transform = DataAugmentationDINO(
        global_crops_scale=(0.4, 1.0), 
        local_crops_scale=(0.1, 0.4), 
        local_crops_number=local_crops_number
    )

    refs_loader = get_epill_dataloader('refs', batch_size=1,crop_transforms=crop_transform)


    for i, data in enumerate(refs_loader):
        image = data['image']
        global_crops = data['global_crops']
        local_crops = data['local_crops']
        break
    
    # print(image.squeeze(dim=0).shape)
    # simple_imshow_tensor(image.squeeze(dim=0))
    # for crop in global_crops:
    #     simple_imshow_tensor(crop.squeeze(dim=0))
    # for crop in local_crops:
    #     simple_imshow_tensor(crop.squeeze(dim=0))
