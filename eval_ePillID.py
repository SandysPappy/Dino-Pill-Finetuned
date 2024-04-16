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
                        default="./dino/pretrained/dino_deitsmall8_pretrain_full_checkpoint.pth",
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


    dinov1_model = initDinoV1Model(model_to_load=FLAGS.dino_base_model_weights,FLAGS=FLAGS,checkpoint_key="teacher", use_back_bone_only=False)

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

    #ref_data = get_epill_dataloader('refs', FLAGS.batch_size, True)
    holdout_data = get_epill_dataloader('holdout', FLAGS.batch_size, True)

    # extract feature
    save_feature_path = "datasets/ref_feature/"
    ref_feature = {}
    '''
    for batch in ref_data:
        image = batch['image']
        label = batch['label']
        image = image.to("cuda")
        feature = dinov1_model(image)
        feature = feature.to("cpu") 
        label = label.to("cpu")
        print(label.device)
        label = label.tolist()
        
        print(feature.device)
        for i in range(FLAGS.batch_size):
            if label[i] not in ref_feature:
                ref_feature[label[i]] = [None, None]
            if batch['is_front'][i] == 'True':
                ref_feature[label[i]][1]=feature[i]
            else:
                ref_feature[label[i]][0]=feature[i]            
    
    print("===ref feature===")
    print(ref_feature)
    print(len(ref_feature))
    '''
    
    holdout_features = []
    holdout_labels = []
    i = 0
    
    for batch in holdout_data:
        if i == 1:
            break
        print(torch.cuda.memory_allocated())
        image = batch['image']
        label = batch['label']
        image = image.to("cuda")
        feature = dinov1_model(image).clone()
        feature = feature.to("cpu")
        label = label.to("cpu")
        label = label.tolist()
        
        for x in feature:
            holdout_features.append(x)
        for x in label:
            holdout_labels.append(x)
        i += 1
    print("===holdout feature===")
    print(holdout_features)
    print(len(holdout_features))
    print(holdout_labels)
    print(len(holdout_labels)) 










    





