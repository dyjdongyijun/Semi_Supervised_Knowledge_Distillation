import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class TinyImageNet(Dataset):
    def __init__(self, root, indexs, split='train', 
                 transform=None, target_transform=None,
                 download=False, return_idx=False):
        self.return_idx = return_idx
        self.indices = indexs
        self.root = root
        self.split = split
        self.transform = transform
        
        self.images, self.labels = self._load_images_labels()
        if indexs is not None:
            self.images = self.images[indexs]
            self.targets = np.array(self.targets)[indexs]

    def _load_images_labels(self):
        images = []
        labels = []
        data_dir = os.path.join(self.root, self.split)

        for label, class_name in enumerate(os.listdir(data_dir)):
            class_dir = os.path.join(data_dir, class_name, 'images')
            for image_name in os.listdir(class_dir):
                images.append(os.path.join(class_dir, image_name))
                labels.append(label)

        return images, labels

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)
