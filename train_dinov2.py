from utils import utils
from utils.DinoModel import DinoModel, dino_args
from utils.utils import NpEncoder
import torch
import argparse
import time 
import faiss
import numpy as np
import json
import os
import torchvision
import torchvision.transforms as T
import torch.distributed as dist
from torchvision.io import read_image
from PIL import Image 
from dataset_loaders import get_epill_dataloader
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
import torch.nn as nn
import torch.nn.functional as F
from utils import metrics


# taken from https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/loss/dino_clstoken_loss.py#L13
class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        self.apply_center_update()
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        return Q.t()

    def forward(self, student_output_list, teacher_out_softmaxed_centered_list):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # TODO: Use cross_entropy_distribution here
        total_loss = 0
        for s in student_output_list:
            lsm = F.log_softmax(s / self.student_temp, dim=-1)
            for t in teacher_out_softmaxed_centered_list:
                loss = torch.sum(t * lsm, dim=-1)
                total_loss -= loss.mean()
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_output * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True
        
if __name__=="__main__":
    parser = argparse.ArgumentParser('DinoV2 args')
    parser.add_argument('--learning_rate',
                        type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=32,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='DINO',
                        help='Directory to put logging.')
    parser.add_argument('--mode',
                        type=str,
                        default="train",
                        help='type of mode train or test')
    parser.add_argument('--seed', 
                        default=0, 
                        type=int, 
                        help='Random seed.')
    parser.add_argument('--num_workers', 
                        default=4, 
                        type=int, 
                        help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", 
                        default="env://", 
                        type=str, 
                        help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", 
                        default=0, 
                        type=int, 
                        help="Please ignore and do not set this argument.")

    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)

    utils.init_distributed_mode(args=FLAGS)

    TEST_SPLIT_FOR_ZERO_SHOT_RETRIEVAL = 0.7
    SEED_FOR_RANDOM_SPLIT = 43

    # vits14 vitb14 vitl14 vitg14
    backbone_arch = "vits14" # change this to vitb14 when ready to really train
    backbone_name = f"dinov2_{backbone_arch}"

    dinov2_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    print(dinov2_model)
    print(dinov2_model.head)
    dinov2_model.eval()
    dinov2_model.cuda()
    
    ref_data = get_epill_dataloader('refs', FLAGS.batch_size, use_dinov1_norm=True)
    holdout_data = get_epill_dataloader('holdout', FLAGS.batch_size, use_dinov1_norm=True)

    # extract feature
    print("start extracting feature")
    feature_path = "features/"
    ref_features = []
    ref_labels = []
    
    for batch in tqdm(ref_data):
        images = batch['image']
        labels = batch['label']
        images = images.to("cuda")
        features = dinov2_model(images)

        break 
        for x in features:
            x = torch.Tensor(x)
            x = F.normalize(x, dim=0)
            ref_features.append(x)
        
        for x in labels:
            ref_labels.append(x)
    





