from torch.utils.data import DataLoader
from dataset_builders import get_epill_dataset

def get_epill_dataloader(fold=None, batch_size=None, use_epill_transforms=None, use_dinov1_norm=None, crop_transforms=None):
    if fold == None:
        raise KeyError("Please insert which fold to use")
    if batch_size == None:
        raise KeyError("Please insert batch size")
    
    dataset = get_epill_dataset(fold, use_epill_transforms=use_epill_transforms, use_dinov1_norm=use_dinov1_norm, crop_transforms=crop_transforms)
    loader = DataLoader(dataset, batch_size=batch_size)
    return loader