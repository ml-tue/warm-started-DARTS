import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import copy
from model_search import Network
import torchvision
import torchvision.transforms as transforms
from datasets.aircraft import FGVCAircraft
import datasets.tinyimagenet as tinyimagenet
import datasets.my_subset as my_subset
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import collections

from datasets.my_subset import Subset2
from datasets.datasets2 import datasets2 
from t2v.task2vec import Task2Vec
from t2v.models import get_model
from t2v.datasets import get_dataset
import t2v.task_similarity as task_similarity

parser = argparse.ArgumentParser("task2vec")
parser.add_argument('--workers', type=int, default=8, help='number of workers to load dataset')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='Select dataset: CIFAR10, CIFAR100, MNIST, FashionMNIST, SVHN')
args = parser.parse_args()

dataset_names = ['aircraft','dtd', 'birds', 'flower', 'imagenet',]

def main():
    
    """
    #  prepare dataset
    if args.dataset == 'omniglot':
        train_data = omniglot(args, "train")
        val_data = omniglot(args, "val")
    elif args.dataset == 'Aircraft':
        train_transform, val_transform = utils._data_transforms(args, "Aircraft")
        train_data = FGVCAircraft(root="/home/grobi/pdarts-v2/data/aircraft", split="train", download=True, transform=train_transform)
        val_data = FGVCAircraft(root="/home/grobi/pdarts-v2/data/aircraft", split="val", download=True, transform=val_transform)
    elif args.dataset == 'ImageNet':
        train_data = imagenet(args, "train")
        val_data = imagenet(args, "val")
    elif args.dataset == 'dtd':
        train_data = dtd(args, "train")
        val_data = dtd(args, "val")
    elif args.dataset == 'birds':
        train_data = birds(args, "train")
        val_data = birds(args, "val")
    elif args.dataset == 'flower':
        train_data = flower(args, "train")
        val_data = flower(args, "val")
    else:
        raise ValueError(args.dataset)
    """

    embeddings = []
    for name in dataset_names:
        print(f"Embedding {name}")
        if name == 'aircraft':
            DATASET_CLASSES = 102
        elif name == 'imagenet':
            DATASET_CLASSES = 200
        elif name == 'dtd':
            DATASET_CLASSES = 47
        elif name == 'birds':
            DATASET_CLASSES = 200
        elif name == 'flower':
            DATASET_CLASSES = 102
        else:
            raise ValueError(args.dataset)
        probe_network = get_model('resnet34', pretrained=True, num_classes=DATASET_CLASSES).cuda()
        embeddings.append( Task2Vec(probe_network, method='montecarlo').embed(datasets2(name)) )
    
    print(embeddings)
    print("-----------------------")
    task_similarity.plot_distance_matrix(embeddings, dataset_names)


if __name__ == '__main__':
    main()