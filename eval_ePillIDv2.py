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
import torch.nn.functional as F
from utils import metrics

def initDinoV1Model(model_to_load, FLAGS, checkpoint_key="teacher", use_back_bone_only=False):
    dino_args.pretrained_weights = model_to_load
    dino_args.output_dir = FLAGS.log_dir
    dino_args.checkpoint_key = checkpoint_key
    dino_args.use_cuda = torch.cuda.is_available()
    dinov1_model = DinoModel(dino_args, use_only_backbone=use_back_bone_only)
    dinov1_model.eval()
    return dinov1_model

if __name__=="__main__":


    parser = argparse.ArgumentParser('Efficient DINO')
    parser.add_argument('--learning_rate',
                        type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=4,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='DINO',
                        help='Directory to put logging.')
    parser.add_argument('--mode',
                        type=str,
                        default="train",
                        help='type of mode train or test')
    parser.add_argument('--dino_base_model_weights',
                        type=str,
                        default="./dino/pretrained/dino_vitbase8_pretrain_full_checkpoint.pth",
                        help='dino based model weights')
    parser.add_argument('--dino_custom_model_weights',
                        type=str,
                        default="./weights/dinoxray/checkpoint.pth",
                        help='dino based model weights')
    parser.add_argument('--search_gallery',
                        type=str,
                        default="train",
                        help='dataset in which images will be searched')
    parser.add_argument('--topK',
                        type=int,
                        default=5,
                        help='Top-k paramter, defaults to 5')
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
    backbone_arch = "vitb14"
    backbone_name = f"dinov2_{backbone_arch}"

    dinov2_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    dinov2_model.eval()
    dinov2_model.cuda()
    '''
    data_path = "./data/ePillID_data/classification_data/segmented_nih_pills_224/"
    dinov1_transform = T.Compose([    
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    batch = torch.zeros([FLAGS.batch_size,3,224,224])
    for i in range(FLAGS.batch_size):
        img = Image.open(f"{data_path}42291-81{i}_0_0.jpg")
        #img = read_image(f"{data_path}42291-81{i}_0_0.jpg")
        img = dinov1_transform(img)
        batch[i] = img 
    print("shape:", batch.shape)
    batch = batch.to("cuda")
    pred = dinov1_model(batch)

    print("result:", pred)
    '''
    
    ref_data = get_epill_dataloader('refs', FLAGS.batch_size, True)
    holdout_data = get_epill_dataloader('holdout', FLAGS.batch_size, True)

    # extract feature
    print("start extracting feature")
    feature_path = "features/"
    ref_features = []
    ref_labels = []
    
    for batch in tqdm(ref_data):
        images = batch['image']
        #print("image shape:", images.shape)
        labels = batch['label']
        images = images.to("cuda")
        features = dinov2_model(images)
        features = features.to("cpu")
        features = features.tolist()
        labels = labels.to("cpu")
        labels = labels.tolist()
        
        for x in features:
            x = torch.Tensor(x)
            x = F.normalize(x, dim=0)
            ref_features.append(x)
        
        for x in labels:
            ref_labels.append(x)
    
    torch.save(ref_features, feature_path+"ref_features_backbone_v2.pt")
    #print("loading ref_features...")
    #ref_features = torch.load(feature_path+"ref_features.pt")    
    
    holdout_features = []
    holdout_labels = []
 
    for batch in tqdm(holdout_data):
        
        images = batch['image']
        labels = batch['label']
        images = images.to("cuda")
        features = dinov2_model(images)
        features = features.to("cpu")
        features = features.tolist()
        labels = labels.to("cpu")
        labels = labels.tolist()
         
        for x in features:
            x = torch.Tensor(x)
            x = F.normalize(x, dim=0)
            holdout_features.append(x)
        
        for x in labels:
            holdout_labels.append(x)
    
    torch.save(holdout_features, feature_path+"holdout_features_backbone_v2.pt")
    #print("loading holdout_features...")
    #holdout_features = torch.load(feature_path+"holdout_features.pt")
    # calculate cosine similarity
    
    print("calculate cosine similarity")
    predict_list = []
    
    for i in tqdm(range(len(holdout_features))):
        max_cos=0
        max_label=-1
        for j in range(len(ref_features)):
            a = holdout_features[i]
            a = a.to("cuda")
            b = ref_features[j]
            b = b.to("cuda")
            #print("a shape:", a.shape)
            #print("b shape:", b.shape)
            cos = F.cosine_similarity(a, b, dim=0)
            if cos > max_cos:
                max_cos = cos
                max_label = ref_labels[j]
            #tup = ref_labels[j], cos
            #cos_list.append(tup)
        #sorted_cos_list = sorted(cos_list, key=lambda x: x[1], reverse=True)
        predict_list.append(max_label)
    torch.save(predict_list, "predict_list_backbone_only_v2.pt")
    print("====predict_list====")
    print("len:", len(predict_list))
    print(predict_list)
    
    #predict_list = torch.load("predict_list_backbone_only.pt")
    c = 0
    for i in range(len(holdout_labels)):
        if holdout_labels[i] == predict_list[i]:
            print("match")
            c+=1
    print("c:", c)
    a_list =[]
    p_list =[]
    for i in holdout_labels:
        a_list.append([i])
    for i in predict_list:
        p_list.append([i])
    Map_result = metrics.mapk(a_list, p_list)
    print("MAP score:", Map_result)
    
    


    





