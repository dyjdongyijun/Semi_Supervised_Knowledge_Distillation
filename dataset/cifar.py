import logging
import math


import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC


import os
import torch
from apricot import FacilityLocationSelection
import IPython
import time

logger = logging.getLogger(__name__)

cifar10_config = {
    'mean': (0.4914, 0.4822, 0.4465),
    'std': (0.2471, 0.2435, 0.2616),
    'size': 32,
    'ncls': 10,
}

cifar100_config = {
    'mean': (0.5071, 0.4867, 0.4408),
    'std': (0.2675, 0.2565, 0.2761),
    'size': 32,
    'ncls': 100,
}


def get_offline_teacher(args):
    pretrained_name = os.path.join(args.pretrain_path, args.dataset, f'{args.teacher_arch}_{args.teacher_data}{args.teacher_dim:d}.pt')
    teacher = torch.load(pretrained_name)[:args.num_train]
    return teacher.float()


def train_val_transforms(data_config):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=data_config['size'],
                              padding=int(data_config['size']*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config['mean'], std=data_config['std'])
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config['mean'], std=data_config['std'])
    ])
    return transform_labeled, transform_val


def get_cifar10(args, root):
    transform_labeled, transform_val = train_val_transforms(cifar10_config)
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled, 
    )

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(
            mean=cifar10_config['mean'], 
            std=cifar10_config['std'], 
            size=cifar10_config['size'], 
            m=args.augstrength), 
        return_idx=True,
    )

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False,
    )

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args, root):
    transform_labeled, transform_val = train_val_transforms(cifar100_config)
    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled,
        return_idx=True,
    )

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(
            mean=cifar100_config['mean'], 
            std=cifar100_config['std'], 
            size=cifar100_config['size'],
            m=args.augstrength),
        return_idx=True,
    )

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False,
    )

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.arange(labels.size)
    if args.percentunl < 100.:
        unlabeled_idx = np.load(os.path.join(f"dataset/inds_{args.dataset}_percentunl_{args.percentunl}.npy"))
    
    # label selection
    if args.labeler=='unif':
        labeled_idx = np.random.choice(labels, args.num_labeled, False)
    elif args.labeler=='class':
        labeled_idx = []
        for i in range(args.num_classes):
            idx = np.where(labels == i)[0]
            idx = np.random.choice(idx, label_per_class, False)
            labeled_idx.extend(idx)
        labeled_idx = np.array(labeled_idx)
    elif args.labeler.split('-')[0] == 'active':
        teacher = get_offline_teacher(args)
        fname = os.path.join(args.pretrain_path, args.dataset, f'{args.teacher_arch}_{args.teacher_data}{args.teacher_dim:d}_{args.labeler}_{args.num_labeled}.npy')
        print(f"Checking if have previously cached results for args.labeler = {args.labeler} at: \n\t{fname}")
        
        found = False
        try:
            labeled_idx = np.load(fname)
            print("\tFOUND")
            found = True
        except:
            print(f"Could not find saved labeled indices at {fname}, proceeding to compute and save accordingly...")
        
        if not found:
            if args.labeler.split('-')[1] == 'ls':
                # leverage scores 
                row_norms = (teacher * teacher).sum(dim=1)
                labeled_idx = np.random.choice(labels, args.num_labeled, False, p=row_norms/row_norms.sum())

            elif args.labeler.split('-')[1] == 'fl':
                # vopt (facility location)
                tic = time.time()
                selector = FacilityLocationSelection(args.num_labeled, metric='cosine', optimizer='stochastic')
                selector.fit(teacher.detach().numpy())
                toc = time.time()
                print(f"\tTime to select subset via {args.labeler} = {toc - tic}")
                labeled_idx = selector.ranking

            else:
                raise Exception(f'args.labeler = {args.labeler} not found')

            # save labeled indices 
            np.save(fname, labeled_idx)
            print("\tsaved!")
        
        
    else:
        raise Exception(f'args.labeler = {args.labeler} not found')
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    def __init__(self, mean, std, size, m=10):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size,
                                  padding=int(size*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size,
                                  padding=int(size*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=m)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, return_idx=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.return_idx = return_idx
        self.indices = indexs
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_idx:
            return img, target, self.indices[index]
        else:
            return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, return_idx=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.return_idx = return_idx
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_idx:
            return img, target, index
        else:
            return img, target

