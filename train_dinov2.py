import argparse
import glob
import os
import pickle
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

sys.path.append("./../dino")
sys.path.append("./../src")

from dinov2.dino import utils

from dinov2.src.constants import COLORS
from dinov2.src.plotting.vis_tools import visualize_attention

from dataset_loaders import get_epill_dataloader

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, image_size=224):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(int(96/(244/image_size)), scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
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

# Call the above function to seed everything
seed_everything()
local_crops_number = 9
batch_size = 32

crop_transform = DataAugmentationDINO(
    global_crops_scale=(0.4, 1.0), 
    local_crops_scale=(0.05, 0.4), 
    local_crops_number=local_crops_number
)

refs_loader = get_epill_dataloader('refs', batch_size=6, crop_transforms=crop_transform)

for i, data in enumerate(refs_loader):
    image = data['image']
    global_crops = data['global_crops']
    local_crops = data['local_crops']

    print(image.shape)
    
    print(f'{len(global_crops)} global crops')
    for crop in global_crops:
        print(crop.shape)

    print(f'{len(local_crops)} local crops')
    for crop in local_crops:
        print(crop.shape)

    break