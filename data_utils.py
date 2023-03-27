import logging
from PIL import Image
import os

import torch

from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from dataset import Blood_Data
# from autoaugment import AutoAugImageNetPolicy
# from dataset_her import CubDataset
import torchvision
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def get_loader(args):

    if args.dataset == 'blood_17':
        train_transform = transforms.Compose([transforms.Resize((240, 240)),
                                              transforms.RandomCrop((224, 224)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((240, 240)),
                                             transforms.CenterCrop((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = Blood_Data('/data1/zhangsc/Projects/TransFG/train.txt',transform=train_transform)
        testset = Blood_Data('/data1/zhangsc/Projects/TransFG/test.txt',transform=test_transform)
    elif args.dataset == 'blood_15':
        train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                              # transforms.RandomCrop((224, 224)),
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                             # transforms.CenterCrop((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = Blood_Data('/data2/Public_dataset/Peripheral_blood_15/train.txt',transform=train_transform,mode='train')
        testset = Blood_Data('/data2/Public_dataset/Peripheral_blood_15/test.txt',transform=test_transform,mode='test')
    else:
        train_transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0), (8, 8, 8, 8), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        test_transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = Blood_Data('/data1/syx/dataset/cell_anno_1115_global_text_jpg_split/cell_anno_1115_global_text_train.txt', transform=train_transform,
                              mode='train')
        testset = Blood_Data('/data1/syx/dataset/cell_anno_1115_global_text_jpg_split/cell_anno_1115_global_text_test.txt', transform=test_transform,
                             mode='test')
    # train_sampler = RandomSampler(trainset) #if args.local_rank == -1 else DistributedSampler(trainset)
    # test_sampler = SequentialSampler(testset) #if args.local_rank == -1 else DistributedSampler(testset)
    train_loader = DataLoader(trainset,
                              # sampler=train_sampler,
                              shuffle=True,
                              batch_size=args.batch_size,
                              num_workers=8,
                              drop_last=True)
    test_loader = DataLoader(testset,
                             # sampler=test_sampler,
                             shuffle=False,
                             batch_size=args.batch_size,
                             num_workers=8,
                             drop_last=False) if testset is not None else None

    return train_loader, test_loader
