from myutils import utils
from myutils.DinoModel import DinoModel, dino_args
from myutils.utils import NpEncoder
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
from myutils import metrics
from torch.utils.data import DataLoader
from dataset_builders_2 import get_epill_dataset
from dino.vision_transformer import vit_small
from empty_model import MLP
def initDinoV1Model(model_to_load, FLAGS, checkpoint_key="teacher", use_back_bone_only=True):
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
                        type=int, default=24,
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
                        default="./dinov1_train_result_new/checkpoint.pth",
                        help='dino based model weights')
    parser.add_argument('--search_gallery',
                        type=str,
                        default="train",
                        help='dataset in which images will be searched')
    parser.add_argument('--topK',
                        type=int,
                        default=50,
                        help='Top-k paramter, defaults to 5')
    parser.add_argument('--seed', 
                        default=0, 
                        type=int, 
                        help='Random seed.')
    parser.add_argument('--num_workers', 
                        default=1, 
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


    dinov1_model = initDinoV1Model(model_to_load=FLAGS.dino_base_model_weights,FLAGS=FLAGS,checkpoint_key="teacher")
    #dinov1_model = vit_small(patch_size=17).to("cuda")
    #dinov1_model = MLP().to("cuda")
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

    dataset = get_epill_dataset('refs', use_epill_transforms=None, use_dinov1_norm=True, crop_transforms=None, do_ocr = False)
    test_dataset = get_epill_dataset('holdout', use_epill_transforms=None, use_dinov1_norm=True, crop_transforms=None, do_ocr=False) 

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_query = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        pin_memory=True,
        drop_last=False,
    )
 
    
    #ref_data = get_epill_dataloader('refs', FLAGS.batch_size, use_epill_transforms=None, use_dinov1_norm=True, crop_transforms=None)
    #holdout_data = get_epill_dataloader('holdout', FLAGS.batch_size, use_epill_transforms=None, use_dinov1_norm=True, crop_transforms=None)

    # extract feature
    print("start extracting feature")
    '''
    feature_path = "features/"
    ref_features = []
    ref_labels = []
    
    for batch in tqdm(ref_data):
        images = batch['image']
        #print("image shape:", images.shape)
        labels = batch['label']
        images = images.to("cuda")
        features = dinov1_model(images)
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
    #torch.save(ref_labels, "ref_labels.pt") 
    torch.save(ref_features, feature_path+"ref_features_finetune.pt")
    #print("loading ref_features...")
    #ref_features = torch.load(feature_path+"ref_features_backbone.pt.pt")    
    
    holdout_features = []
    holdout_labels = []
 
    for batch in tqdm(holdout_data):
        
        images = batch['image']
        labels = batch['label']
        images = images.to("cuda")
        features = dinov1_model(images)
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
    #torch.save(holdout_labels, "holdout_labels.pt")
    torch.save(holdout_features, feature_path+"holdout_features_finetune.pt")
    #print("loading holdout_features...")
    #holdout_features = torch.load(feature_path+"holdout_features_backbone.pt")
    # calculate cosine similarity
    '''

    ref_labels = dataset.ids
    ref_info = dataset.labels
    #ref_ocr = dataset.ocr_res
    print("len ref_labels:", len(ref_labels))
    holdout_labels = test_dataset.ids
    holdout_info = test_dataset.labels
    #holdout_ocr = test_dataset.ocr_res
    print("len holdout_labels:", len(holdout_labels))

    dataset.extract_features(dinov1_model,data_loader=data_loader_train)
    test_dataset.extract_features(dinov1_model,data_loader=data_loader_query)

    ref_features = []
    holdout_features = []
    for i in range(len(dataset)):
        if ref_labels[i] in holdout_labels:
            ref_features.append(dataset.image_features[i])
    for i in range(len(test_dataset)):
        holdout_features.append(test_dataset.image_features[i])
    print("ref len",len(ref_features) )
    print("hold out", len(holdout_features))

    ref_features = torch.from_numpy(np.array(ref_features))
    holdout_features = torch.from_numpy(np.array(holdout_features))

    ref_features = ref_features.reshape(ref_features.size(0), -1)
    holdout_features = holdout_features.reshape(holdout_features.size(0), -1) 

    '''
    print("calculate cosine similarity")
    predict_list = []
    
    for i in tqdm(range(len(holdout_features))):
        max_cos=0
        max_label=-1
        cos_list=[]
        for j in range(len(ref_features)):
            a = holdout_features[i]
            a = a.to("cuda")
            b = ref_features[j]
            b = b.to("cuda")
            #print("a shape:", a.shape)
            #print("b shape:", b.shape)
            cos = F.cosine_similarity(a, b, dim=0)
            #if cos > max_cos:
            #    max_cos = cos
            #    max_label = ref_labels[j]
            tup = ref_labels[j][1], cos
            cos_list.append(tup)
        sorted_cos_list = sorted(cos_list, key=lambda x: x[1], reverse=True)

        temp_list=[]
        for n in range(FLAGS.topK):
            temp_list.append(sorted_cos_list[n][0])
        predict_list.append(temp_list)
    torch.save(predict_list, "predict_list_backbone_only.pt")
    print("====predict_list====")
    print("len:", len(predict_list))
    print(predict_list)
    
    #predict_list = torch.load("predict_list_backbone_only.pt")
    for i in predict_list:
        print("predict:", i)
    c = 0
    for i in range(len(holdout_labels)):
        if holdout_labels[i][1] in predict_list[i]:
            print("match")
            c+=1
    print("c:", c)
    '''
    d = ref_features.size(-1)    # dimension
    nb = ref_features.size(0)    # database size
    nq = holdout_features.size(0)      # nb of queries

    index = faiss.IndexFlatL2(d)   # build the index
    print(index.is_trained)
    index.add(ref_features)    # add vectors to the index
    print(index.ntotal)

    topK  = FLAGS.topK
    k = FLAGS.topK                          # we want to see 4 nearest neighbors
    #D, I = index.search(gallery_features[:5], k) # sanity check
    #print(I)
    #print(D)
    D, I = index.search(holdout_features, k)     # actual search

    a_list = []
    a_img = []
    p_list = []
    p_img = []
    lcID_list = []
    pcID_list = []
    #print("I:", I)
    #print("len I", len(I))
    
    for i in I:
        temp=[]
        #temp_lc=[]
        #temp_pc=[]
        for idx in i:
            temp.append(ref_labels[idx])
            p_img.append(ref_info[idx][7])
            #temp_lc.append(ref_info[idx][2])
            #temp_pc.append(ref_info[idx][3])
        p_list.append(temp)
        #lcID_list.append(temp_lc)
        #pcID_list.append(temp_pc)
    for i in holdout_labels:
        a_list.append([i])
    #for i in predict_list:
    #    p_list.append([i])
    #p_list = predict_list
    '''
    merged_list = []
    for i in range(745):
        query_lc = lcID_list[i]
        query_pc = pcID_list[i]
        match = []
        unmatch = []
        for j in range(len(query_lc)):
            if query_lc[j] == holdout_ocr[i] or query_pc[j] == holdout_ocr[i]:
                match.append(p_list[i][j])
            else:
                unmatch.append(p_list[i][j])
            merge = match + unmatch
            merge = merge[:FLAGS.topK]
        merged_list.append(merge)
    '''
    c = 0
    for i in range(745):
        print("p_list[i]:", p_list[i], "img:", p_img[i])
        print("a_list[i]:", a_list[i], "img", holdout_info[i][7])
        if p_list[i][0] == a_list[i][0]:
            c+=1
    print("c:", c)
    print("merged_list len:", len(p_list))
    print("a_list len:", len(a_list))
    
    Map_result = metrics.mapk(a_list, p_list)
    print("MAP score:", Map_result) 

    print("ref len",len(ref_features) )
    print("hold out len", len(holdout_features))




