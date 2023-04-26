import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

fashionmnist_mean = (0.5,)
fashionmnist_std = (0.5,)
fashionmnist_mean, fashionmnist_std = (0.1307,), (0.3081,)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_fashionmnist(args, root):
    transform_labeled = transforms.Compose([
        transforms.Lambda(lambda x: x[None,:,:].repeat(3, 1, 1)),
        transforms.ToPILImage(mode='RGB'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=28,
                              padding=4,
                              padding_mode='reflect'),
        transforms.ToTensor(),
#         
        transforms.Normalize(mean=fashionmnist_mean, std=fashionmnist_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=fashionmnist_mean, std=fashionmnist_std)
    ])
    base_dataset = datasets.FashionMNIST(root, train=True, download=True)
    

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = FashionMNISTSSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled, 
    )
    
    
    
    train_unlabeled_dataset = FashionMNISTSSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=fashionmnist_mean, std=fashionmnist_std), 
        return_idx=True,
    )
    
    
    test_dataset = datasets.FashionMNIST(
        root, train=False, transform=transform_val, download=True,
    )

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset




def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    
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
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.Lambda(lambda x: x[None,:,:].repeat(3, 1, 1)),
            transforms.ToPILImage(mode='RGB'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=28,
                                  padding=4,
                                  padding_mode='reflect'),
        ])
        
        self.strong = transforms.Compose([
            transforms.Lambda(lambda x: x[None,:,:].repeat(3, 1, 1)),
            transforms.ToPILImage(mode='RGB'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=28,
                                  padding=4,
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class FashionMNISTSSL(datasets.FashionMNIST):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True, return_idx=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        
#         self.data = self.data.float() / 255.
#         self.data = self.data[:,None,:,:]
        self.return_idx = return_idx
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.indices = np.arange(len(self.targets))[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        #print(type(img), type(target))
        #img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_idx:
            return img, target, self.indices[index]
        else:
            return img, target

