# data source: wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from .cifar import train_val_transforms, x_u_split, TransformFixMatch

tiny_imagenet_config = {
    'mean': (0.4802, 0.4481, 0.3975),
    'std': (0.2302, 0.2265, 0.2262),
    'size': 64,
    'ncls': 200,
}


def get_tiny_imagenet(args, root):
    data_root = os.path.join(root, 'tiny-imagenet-200')
    
    transform_labeled, transform_val = train_val_transforms(tiny_imagenet_config)
    base_dataset = TinyImageNet(data_root, split='train')

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.labels)

    train_labeled_dataset = TinyImageNet(
        data_root, train_labeled_idxs, split='train', 
        transform=transform_labeled, 
    )

    train_unlabeled_dataset = TinyImageNet(
        data_root, train_unlabeled_idxs, split='train', 
        transform=TransformFixMatch(
            mean=tiny_imagenet_config['mean'], 
            std=tiny_imagenet_config['std'], 
            size=tiny_imagenet_config['size'], 
            m=args.augstrength), 
        return_idx=True,
    )

    test_dataset = TinyImageNet(data_root, split='val', transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset



class TinyImageNet(Dataset):
    def __init__(self, root, indexs=None, split='train', 
                 transform=None, target_transform=None,
                 download=False, return_idx=False):
        self.return_idx = return_idx
        self.indices = indexs
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        self.images, self.labels = self._load_images_labels()
        if indexs is not None:
            self.images = [self.images[i] for i in indexs]
            self.labels = np.array(self.labels)[indexs]

    def _load_images_labels(self):
        images, labels = [], []
        data_dir = os.path.join(self.root, self.split)

        if self.split == 'train':
            for label, class_name in enumerate(os.listdir(data_dir)):
                class_dir = os.path.join(data_dir, class_name, 'images')
                for image_name in os.listdir(class_dir):
                    images.append(os.path.join(class_dir, image_name))
                    labels.append(label)
        else:  # 'val' split
            class_name_to_label = {class_name: i for i, class_name in enumerate(os.listdir(os.path.join(self.root, 'train')))}
            val_annotations_path = os.path.join(self.root, 'val', 'val_annotations.txt')
            with open(val_annotations_path) as f:
                for line in f.readlines():
                    tokens = line.split('\t')
                    image_name, class_name = tokens[0], tokens[1]
                    images.append(os.path.join(data_dir, 'images', image_name))
                    labels.append(class_name_to_label[class_name])

        return images, labels

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
        
        if self.return_idx:
            return image, label, self.indices[index]
        else:
            return image, label

    def __len__(self):
        return len(self.images)
