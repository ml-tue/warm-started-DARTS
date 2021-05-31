from datasets.tinyimagenet import ImageFolder
import torchvision.datasets as dset
from sklearn.model_selection import train_test_split
import torch.utils
import utils2 as utils
import numpy as np
import torch
from datasets.my_subset import Subset2
from datasets.aircraft import FGVCAircraft
from torch.utils.data import ConcatDataset

def datasets2(dt):
    if dt == "imagenet":
        return imagenet("all")
    elif dt == "dtd":
        return dtd("all")
    elif dt == "aircraft":
        return aircraft("trainval")
    elif dt == "birds":
        return birds("all")
    elif dt == "flower":
        return flower("all")
    elif dt == "dtd":
        return dtd("all")
    elif dt == "omniglot":
        return omniglot("all")


def aircraft(split="train"):
    train_transform, val_transform = utils._data_transforms("Aircraft")
    train_data = FGVCAircraft(root="/home/grobi/pdarts-v2/data/aircraft", split="trainval", download=True, transform=val_transform)
    val_data = FGVCAircraft(root="/home/grobi/pdarts-v2/data/aircraft", split="test", download=True, transform=val_transform)
    
    if split == "trainval":
        return train_data
    elif split == "test":
        return val_data
    elif split == "all":
        return train_data, val_data
    else:
        raise ValueError("Split not deffined")


def imagenet (split="train"):
    train_transform, val_transform = utils._data_transforms("ImageNet")
    dataset = ImageFolder(root="/home/grobi/pdarts-v2/data/tiny-imagenet/train", transform=val_transform)
    
    targets = dataset.targets
    trainval_idx, test_idx= train_test_split(
        np.arange(len(targets)), 
        test_size=0.2,
        shuffle=True,
        random_state=1,
        stratify=targets,
    )

    targets2 = [dataset.targets[idx] for idx in trainval_idx]
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=0.5,
        shuffle=True,
        random_state=1,
        stratify=targets2,
    )

    if split == "trainval":
        return Subset2(dataset, trainval_idx, val_transform)
    elif split == "test":
        return Subset2(dataset, test_idx, val_transform)
    elif split == "all":
        return dataset
    else:
        raise ValueError("Split not deffined")


def dtd (split="train"):
    train_transform, val_transform = utils._data_transforms("dtd")
    dataset = dset.ImageFolder(root="/home/grobi/pdarts-v2/data/dtd/images", transform=val_transform)
    
    targets = dataset.targets
    trainval_idx, test_idx= train_test_split(
        np.arange(len(targets)), 
        test_size=0.2,
        shuffle=True,
        random_state=1,
        stratify=targets,
    )

    targets2 = [dataset.targets[idx] for idx in trainval_idx]
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=0.5,
        shuffle=True,
        random_state=1,
        stratify=targets2,
    )

    if split == "trainval":
        return Subset2(dataset, trainval_idx, val_transform)
    elif split == "test" :
        return Subset2(dataset, test_idx, val_transform)
    elif split == "all" :
        return dataset
    else:
        raise ValueError("Split not deffined")


def birds (split="train"):
    train_transform, val_transform = utils._data_transforms("birds")
    dataset = dset.ImageFolder(root="/home/grobi/pdarts-v2/data/CUB_200_2011/images", transform=val_transform)
    
    targets = dataset.targets
    trainval_idx, test_idx= train_test_split(
        np.arange(len(targets)), 
        test_size=0.2,
        shuffle=True,
        random_state=1,
        stratify=targets,
    )

    targets2 = [dataset.targets[idx] for idx in trainval_idx]
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=0.5,
        shuffle=True,
        random_state=1,
        stratify=targets2,
    )

    if split == "trainval":
        return Subset2(dataset, trainval_idx, val_transform)
    elif split == "test" :
        return Subset2(dataset, test_idx, val_transform)
    elif split == "all" :
        return dataset
    else:
        raise ValueError("Split not deffined")


def flower (split="train"):
    train_transform, val_transform = utils._data_transforms("flower")
    dataset = dset.ImageFolder(root="/home/grobi/pdarts-v2/data/flower_data/images", transform=val_transform)
    
    targets = dataset.targets
    trainval_idx, test_idx= train_test_split(
        np.arange(len(targets)), 
        test_size=0.2,
        shuffle=True,
        random_state=1,
        stratify=targets,
    )

    targets2 = [dataset.targets[idx] for idx in trainval_idx]
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=0.5,
        shuffle=True,
        random_state=1,
        stratify=targets2,
    )

    if split == "trainval":
        return Subset2(dataset, trainval_idx, val_transform)
    elif split == "test" :
        return Subset2(dataset, test_idx, val_transform)
    elif split == "all" :
        return dataset
    else:
        raise ValueError("Split not deffined")

def omniglot (split="train"):
    train_transform, val_transform = utils._data_transforms("omniglot")
    train_data = dset.Omniglot(root="/home/grobi/pdarts-v2/data/omni", background=True, download=True, transform=val_transform)
    test_data = dset.Omniglot(root="/home/grobi/pdarts-v2/data/omni", background=False, download=True, transform=val_transform)
    
    """
    if split == 'train':
        return Subset2(train_data, train_idx, train_transform)
    elif split== "val":
        return Subset2(train_data, val_idx, val_transform)
    """
    if split == "trainval":
        return train_data
    elif split == "test" :
        return test_data
    elif split == "all" :
        return torch.cat((train_data, test_data))
    else:
        raise ValueError("Split not deffined")