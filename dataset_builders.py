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

# set use_epill_transforms=True to transform input image when calling __get__
def get_epill_dataset(fold=None, use_epill_transforms=None, use_dinov1_norm=None, crop_transforms=None):
    if fold == None:
        raise KeyError("Please insert which fold to use")

    path_folds = 'datasets/ePillID_data/folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/'

    # note: pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv is the same file as all_labels.csv
    if fold == 'refs':
        return EPillDataset(path_folds+'pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv', use_epill_transforms, use_dinov1_norm=use_dinov1_norm, crop_transforms=crop_transforms)

    if fold == 'holdout':
        return EPillDataset(path_folds+'pilltypeid_nih_sidelbls0.01_metric_5folds_4.csv', use_epill_transforms, use_dinov1_norm=use_dinov1_norm, crop_transforms=crop_transforms)
    if fold == 'fold_0':
        return EPillDataset(path_folds+'pilltypeid_nih_sidelbls0.01_metric_5folds_0.csv', use_epill_transforms, use_dinov1_norm=use_dinov1_norm, crop_transforms=crop_transforms)
    if fold == 'fold_1':
        return EPillDataset(path_folds+'pilltypeid_nih_sidelbls0.01_metric_5folds_1.csv', use_epill_transforms, use_dinov1_norm=use_dinov1_norm, crop_transforms=crop_transforms)
    if fold == 'fold_2':
        return EPillDataset(path_folds+'pilltypeid_nih_sidelbls0.01_metric_5folds_2.csv', use_epill_transforms, use_dinov1_norm=use_dinov1_norm, crop_transforms=crop_transforms)
    if fold == 'fold_3':
        return EPillDataset(path_folds+'pilltypeid_nih_sidelbls0.01_metric_5folds_3.csv', use_epill_transforms, use_dinov1_norm=use_dinov1_norm, crop_transforms=crop_transforms)


# annotations format
# ['images', 'pilltype_id',            'label_code_id', 'prod_code_id', 'is_ref', 'is_front', 'is_new', 'image_path',                  'label']
# ['0.jpg',  '51285-0092-87_BE305F72', '51285',         '92',           'False',  'False',    'False',  'fcn_mix_weight/dc_224/0.jpg', '51285-0092-87_BE305F72']
class EPillDataset(Dataset):
    def __init__(self, path_labels, use_epill_transforms=None, use_dinov1_norm=None, crop_transforms=None):

        # image will be transformed when called in __getitem__ if use_epill_transforms is set
        # rotates, scales, translates, and (sometimes) shears image
        self.use_epill_transforms = use_epill_transforms
        self.use_dinov1_norm = use_dinov1_norm
        self.crop_transforms = crop_transforms

        self.label_index_keys = None
        self.labels = []
        self.images = []
        self.dinov1_norm = T.Compose([
            #T.Resize((224,224)),
            #T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        # set labels and index keys
        with open(path_labels, 'r') as f:
            csv_reader = csv.reader(f)

            self.label_index_keys = next(csv_reader) 
            for label in csv_reader:
                for i, _ in enumerate(label):
                    if label[i] == 'True':
                        label[i] = True
                    if label[i] == 'False':
                        label[i] = False
                self.labels.append(label)

        # encode the gross strings into integers (0, numclasses-1)
        ids = []
        for label in self.labels:
            ids.append(label[1]) # pilltype_id
        le = LabelEncoder()
        le.fit(ids)
        ids = list(le.transform(ids))
        for i, id in enumerate(ids):
            self.labels[i][1] = id

        # set images as list of torch tensors 
        for label in self.labels:
            imgs_path = 'datasets/ePillID_data/classification_data/'
            img = read_image(imgs_path+label[7])
            #img = Image.open(imgs_path+label[7])
            self.images.append(img)

    def __len__(self):
        assert len(self.images) == len(self.labels), f"Number of images does not match number of labels for {type(self).__name__}"
        return len(self.images)

    def __getitem__(self, idx):
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


    
