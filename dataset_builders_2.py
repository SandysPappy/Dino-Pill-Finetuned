import csv
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as T
from imgaug import augmenters as iaa
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import time
from myutils import utils
import torch.distributed as dist
from torchvision.transforms import ToTensor
from easy_ocr import easyocr_prediction
from tqdm import tqdm

# set use_epill_transforms=True to transform input image when calling __get__
def get_epill_dataset(fold=None, use_epill_transforms=None, use_dinov1_norm=None, crop_transforms=None, do_ocr=False):
    if fold == None:
        raise KeyError("Please insert which fold to use")

    path_folds = 'datasets/ePillID_data/folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/'

    fold_list = ["all", "4", "0", "1", "2", "3"] # ref, holdout, fold_0, fold_1, fold_2, fold_3
    folds = []
    for i in range(len(fold_list)):
        folds.append( EPillDataset(f'{path_folds}pilltypeid_nih_sidelbls0.01_metric_5folds_{fold_list[i]}.csv', use_epill_transforms=use_epill_transforms, use_dinov1_norm=use_dinov1_norm, crop_transforms=crop_transforms, do_ocr=do_ocr) )
    #print("len folds:", len(folds))
    all_labels = []
    for i in folds:
        for j in range(len(i.labels)):
            all_labels.append(i.labels[j][1])
    
    le = LabelEncoder()
    le.fit(all_labels)
    encode_labels = list(le.transform(all_labels))

    start = 0
    for i in folds:
        f_len = len(i.labels)
        end = start+f_len
        i.ids = encode_labels[start:end]
        start = end
        #print("f_len:", f_len)
        #print("ids len:", len(i.ids))

    # note: pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv is the same file as all_labels.csv
    if fold == 'refs':
        return folds[0]
    if fold == 'holdout':
        return folds[1]
    if fold == 'fold_0':
        return folds[2]
    if fold == 'fold_1':
        return folds[3]
    if fold == 'fold_2':
        return folds[4]
    if fold == 'fold_3':
        return folds[5]

# annotations format
# ['images', 'pilltype_id',            'label_code_id', 'prod_code_id', 'is_ref', 'is_front', 'is_new', 'image_path',                  'label']
# ['0.jpg',  '51285-0092-87_BE305F72', '51285',         '92',           'False',  'False',    'False',  'fcn_mix_weight/dc_224/0.jpg', '51285-0092-87_BE305F72']
class EPillDataset(Dataset):
    def __init__(self, path_labels, use_epill_transforms=None, use_dinov1_norm=None, crop_transforms=None, do_ocr=False):

        # image will be transformed when called in __getitem__ if use_epill_transforms is set
        # rotates, scales, translates, and (sometimes) shears image
        self.use_epill_transforms = use_epill_transforms
        self.use_dinov1_norm = use_dinov1_norm
        self.crop_transforms = crop_transforms

        #self.img_preprocessing_fn = img_preprocessing_fn
        self.label_index_keys = None
        self.labels = []
        self.images = []
        self.ids = []
        self.image_features= []
        self.ocr_res = []
        self.class_id_to_str = {}
        self.class_str_to_id = {}
        self.dinov1_norm = T.Compose([
            #T.Resize((224,224)),
            #T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        # set labels and index keys
        all_path = 'datasets/ePillID_data/folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv'
        with open(path_labels, 'r') as f:
            csv_reader = csv.reader(f)

            self.label_index_keys = next(csv_reader) 
            for label in csv_reader:
                for i, _ in enumerate(label):
                    if label[i] == 'True':
                        label[i] = True
                    if label[i] == 'False':
                        label[i] = False
                if path_labels==all_path and label[4]==False:
                    continue
                self.labels.append(label)

        # encode the gross strings into integers (0, numclasses-1)
        '''
        for label in self.labels:
            self.ids.append(label[1]) # pilltype_id
        le = LabelEncoder()
        le.fit(self.ids)
        self.ids = list(le.transform(self.ids))
        '''
        '''
        for i, id in enumerate(self.ids):
            if id not in self.class_id_to_str:
                self.class_id_to_str[id] = self.labels[i][1]
            if self.labels[i][1] not in self.class_str_to_id:
                self.class_str_to_id[self.labels[i][1]] = id
            self.labels[i][1] = id
        '''
        # set images as list of torch tensors 
        print("read image and run ocr")
        for label in tqdm(self.labels):
            imgs_path = 'datasets/ePillID_data/classification_data/'
            img = read_image(imgs_path+label[7])
            if path_labels == "datasets/ePillID_data/folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/pilltypeid_nih_sidelbls0.01_metric_5folds_4.csv" and do_ocr==True:
                text = easyocr_prediction(imgs_path+label[7])
                self.ocr_res.append(text)
            #print(text)
            #img = Image.open(imgs_path+label[7]).convert('RGB')
            self.images.append(img)

    def getOriginalImage(self, idx):
        Images = self.images[idx]
        return Images
                        
    def getImagePath(self, idx):
        return None

    def extract_features(self,model, data_loader, use_cuda=True, multiscale=False):
        metric_logger = utils.MetricLogger(delimiter="  ")
        features = None
        # for samples, index in metric_logger.log_every(data_loader, 10):
        for Image_features,labels,image, is_front, index in metric_logger.log_every(data_loader, 10):
            samples = image
            samples = samples.cuda(non_blocking=True)
            index = index.cuda(non_blocking=True)
            if multiscale:
                feats = utils.multi_scale(samples, model)
            else:
                feats = model(samples).clone()

            # init storage feature matrix
            if dist.get_rank() == 0 and features is None:
                features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
                if use_cuda:
                    features = features.cuda(non_blocking=True)
                print(f"Storing features into tensor of shape {features.shape}")

            # get indexes from all processes
            y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
            y_l = list(y_all.unbind(0))
            y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
            y_all_reduce.wait()
            index_all = torch.cat(y_l)

            # share features between processes
            feats_all = torch.empty(
                dist.get_world_size(),
                feats.size(0),
                feats.size(1),
                dtype=feats.dtype,
                device=feats.device,
            )
            output_l = list(feats_all.unbind(0))
            output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
            output_all_reduce.wait()

            # update storage feature matrix
            if dist.get_rank() == 0:
                if use_cuda:
                    features.index_copy_(0, index_all, torch.cat(output_l))
                else:
                    features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
        
        for f in features:
            self.image_features.append(f.cpu().numpy())

        self.isDataTransformed = True  

    def __len__(self):
        assert len(self.images) == len(self.labels), f"Number of images does not match number of labels for {type(self).__name__}"
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx][1] # pilltype_id
        en_label = self.ids[idx]
        is_front = self.labels[idx][5]
        is_ref = self.labels[idx][4]
        img_path = self.labels[idx][7]
        Image_features = []

        #img = EPillDataset.epill_transforms(img)
        img = img.float()
        img = self.dinov1_norm(img)
        #img = self.img_preprocessing_fn(img)

        if len(self.image_features)==len(self):
            Image_features = self.image_features[idx]
        
        return Image_features, en_label, img, is_front, idx 

    def old__getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx][1] # pilltype_id 
        is_front = self.labels[idx][5]
        is_ref = self.labels[idx][4]

        if self.use_dinov1_norm is not None and self.crop_transforms is not None:
            raise KeyError('Using more than one transform. Please only set one type of transform when creating the dataset')

        if self.use_epill_transforms:
            img = EPillDataset.epill_transforms(img)

        # used for training 
        if self.crop_transforms:
            # first 2 crops are global. The rest are local crops
            img , crops = self.crop_transforms(img)
            return {
                "image": img,
                "global_crops": crops[:2],
                "local_crops": crops[2:],
                "label": label,
                "is_front": is_front,
                "is_ref": is_ref
            }

        if self.use_dinov1_norm:
            img = self.dinov1_norm(img)

        return {
            "image": img,
            "label": label,
            "is_front": is_front,
            "is_ref": is_ref
        }
        

    # transforms images according to the epill dataset paper
    @staticmethod
    def epill_transforms(image):
        image = np.array(image)
        affine_seq, ref_seq, cons_seq = EPillDataset.get_imgaug_sequences()
        
        aug_image = affine_seq.augment_images(image)
        aug_image = ref_seq.augment_images(image)
        aug_image = cons_seq.augment_images(image)
        
        return torch.Tensor(aug_image)

    # image transformations used in the paper
    # this is just copy and pasted from here: https://github.com/usuyama/ePillID-benchmark/blob/master/src/image_augmentators.py
    @ staticmethod
    def get_imgaug_sequences(
        low_gblur = 1.0, 
        high_gblur = 3.0,
        addgn_base_ref = 0.01, 
        addgn_base_cons = 0.001,
        rot_angle = 180, 
        max_scale = 1.0,
        add_perspective = False
    ):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
            
        affine_seq = iaa.Sequential([
                iaa.Affine(
                    rotate=(-rot_angle, rot_angle),
                    scale=(0.8, max_scale),
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                ),
                sometimes(iaa.Affine(
                    shear=(-4, 4),
                ))
            ])
        
        affine_list = [affine_seq]

        contrast_list = [
                iaa.Sequential([
                    iaa.LinearContrast((0.7, 1.0), per_channel=False), # change contrast
                    iaa.Add((-30, 30), per_channel=False), # change brightness
                ]),
                iaa.Sequential([
                    iaa.LinearContrast((0.4, 1.0), per_channel=False), # change contrast
                    iaa.Add((-80, 80), per_channel=False), # change brightness
                ])            
            ]

        if add_perspective:
            print("Adding perspective transform to augmentation")
            affine_list =  affine_list + [
                        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                        ]
                                            
            contrast_list = contrast_list + [ 
                iaa.GammaContrast((0.5, 1.7), per_channel=True),
                iaa.SigmoidContrast(gain=(8, 12), cutoff=(0.2,0.8), per_channel=False)
                ]

        ref_seq = iaa.Sequential(affine_list + [
            iaa.OneOf(contrast_list),
            iaa.OneOf([
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 3*addgn_base_ref*255), per_channel=0.5),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, addgn_base_ref*255), per_channel=0.5),
            ]),
            iaa.OneOf([
                iaa.GaussianBlur(sigma=(0, high_gblur)),
                iaa.GaussianBlur(sigma=(0, low_gblur)),
            ])
        ])

        cons_seq = iaa.Sequential(affine_list + [
            iaa.LinearContrast((0.9, 1.1), per_channel=False),
            iaa.Add((-10, 10), per_channel=False),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 5*addgn_base_cons*255), per_channel=0.5),
            iaa.GaussianBlur(sigma=(0, low_gblur)),
        ])
        
        return affine_seq, ref_seq, cons_seq

class EPillDatasetDinov1(EPillDataset):
    def __init__(self, **kwargs):
        super.__init__(**kwargs)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx][1] # pilltype_id
        is_front = self.labels[idx][5]
        is_ref = self.labels[idx][4]

        img = EPillDataset.epill_transforms(img)

        img = self.dinov1_norm(img)

        return img, idx


 
if __name__ == '__main__':
    path_labels = 'datasets/ePillID_data/all_labels.csv'

    dataset_refs = get_epill_dataset('refs', use_epill_transforms=True)
    dataset_fold_0 = get_epill_dataset('fold_0')
    dataset_fold_1 = get_epill_dataset('fold_1', True)
    dataset_fold_2 = get_epill_dataset('fold_2', True)
    dataset_fold_3 = get_epill_dataset('fold_3', True)
    dataset_holdout = get_epill_dataset('holdout', True)

    datasets = [
        dataset_refs,
        dataset_fold_0,
        dataset_fold_1,
        dataset_fold_2,
        dataset_fold_3,
        dataset_holdout
    ]

    # for dataset in datasets:
    #     # checking if all only contains reference images
    #     for labels in dataset:
    #         if labels["is_ref"] == True:
    #             print(labels)

    # for dataset in datasets:
    #     print(len(dataset))
    #     print(dataset[-1])


    
