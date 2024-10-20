import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader, random_split

CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255




def get_num_classes(args):
    if args.dataset == 'cifar10':
        return 10
    elif args.dataset == 'cifar100':
        return 100
    else:
        raise ValueError('Invalid dataset name.')


def get_normalize(args):
    if args.normalize:
        if args.dataset == 'cifar100':
            mu = torch.tensor(CIFAR100_MEAN).view(3,1,1).cuda()
            std = torch.tensor(CIFAR100_STD).view(3,1,1).cuda()
        elif args.dataset == 'cifar10':
            mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
            std = torch.tensor(cifar10_std).view(3,1,1).cuda()
        else:
            raise ValueError("Invalid dataset name")
        normalize = lambda X: (X - mu)/std
    else:
        normalize = lambda X: X
    return normalize

def get_loaders(args, trainDataSet):#改成如果dataset直接是trainDataSet，怎么办
    #trainDataSet的定义是trainDataSet=TensorDataset(torch.tensor(local_data), torch.tensor(local_label))，应该只有数据和标签
    # if args.dataset == 'cifar10':
    #     transform_train = transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #     ])
    #     transform_test = transforms.Compose([transforms.ToTensor()])
    #     trainset = torchvision.datasets.CIFAR10(root='/scratch/CIFAR10', train=True, download=True, transform=transform_train)
    #     testset = torchvision.datasets.CIFAR10(root='/scratch/CIFAR10', train=False, download=True, transform=transform_test)#可能要让这个数据集有train=False的功能？
    #     ospset =  torchvision.datasets.CIFAR10(root='/scratch/CIFAR10', train=True, download=True, transform=transform_train)
    #     subset = list(range(0, args.osp_data_len))
    #     ospset = torch.utils.data.Subset(ospset, subset)
    
    
    subset = list(range(0, args.osp_data_len))
    ospset = torch.utils.data.Subset(trainDataSet, subset)

    train_loader = torch.utils.data.DataLoader(
        dataset=trainDataSet,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    osp_loader = torch.utils.data.DataLoader(
        dataset=ospset,
        batch_size=args.osp_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    return train_loader, osp_loader
