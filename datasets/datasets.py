from datasets.tinyimagenet import ImageFolder
import torchvision.datasets as dset
from sklearn.model_selection import train_test_split
import torch.utils
import utils
import numpy as np
from datasets.my_subset import Subset2

def imagenet (args, split="train"):
    train_transform, val_transform = utils._data_transforms(args, "ImageNet")
    dataset = ImageFolder(root="/home/grobi/pdarts-v2/data/tiny-imagenet/train")
    
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

    if split == 'train':
        return Subset2(dataset, train_idx, train_transform)
    elif split== "val":
        return Subset2(dataset, val_idx, train_transform)
    elif split == "trainval":
        return Subset2(dataset, trainval_idx, train_transform)
    elif split == "test" :
        return Subset2(dataset, test_idx, val_transform)
    else:
        raise ValueError("Split not deffined")


def dtd (args, split="train"):
    train_transform, val_transform = utils._data_transforms(args, "dtd")
    dataset = dset.ImageFolder(root="/home/grobi/pdarts-v2/data/dtd/images")
    
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

    if split == 'train':
        return Subset2(dataset, train_idx, train_transform)
    elif split == "val":
        return Subset2(dataset, val_idx, train_transform)
    elif split == "trainval":
        return Subset2(dataset, trainval_idx, train_transform)
    elif split == "test" :
        return Subset2(dataset, test_idx, val_transform)
    else:
        raise ValueError("Split not deffined")


def birds (args, split="train"):
    train_transform, val_transform = utils._data_transforms(args, "birds")
    dataset = dset.ImageFolder(root="/home/grobi/pdarts-v2/data/CUB_200_2011/images")
    
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

    if split == 'train':
        return Subset2(dataset, train_idx, train_transform)
    elif split == "val":
        return Subset2(dataset, val_idx, train_transform)
    elif split == "trainval":
        return Subset2(dataset, trainval_idx, train_transform)
    elif split == "test" :
        return Subset2(dataset, test_idx, val_transform)
    else:
        raise ValueError("Split not deffined")


def flower (args, split="train"):
    train_transform, val_transform = utils._data_transforms(args, "flower")
    dataset = dset.ImageFolder(root="/home/grobi/pdarts-v2/data/flower_data/images")
    
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

    if split == 'train':
        return Subset2(dataset, train_idx, train_transform)
    elif split == "val":
        return Subset2(dataset, val_idx, train_transform)
    elif split == "trainval":
        return Subset2(dataset, trainval_idx, train_transform)
    elif split == "test" :
        return Subset2(dataset, test_idx, val_transform)
    else:
        raise ValueError("Split not deffined")

def omniglot (args, split="train"):
    train_transform, val_transform = utils._data_transforms(args)
    train_data = dset.Omniglot(root=args.tmp_data_dir, background=True, download=True, transform=train_transform)
    test_data = dset.Omniglot(root=args.tmp_data_dir, background=False, download=True, transform=val_transform)
    
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
    else:
        raise ValueError("Split not deffined")