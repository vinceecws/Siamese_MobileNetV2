from os.path import join
import random
import torch
import time
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import Image
from torch.utils.data import Dataset

class LFW(Dataset): #For now, triplet sampling only
    def __init__(self, data_dir, size):
        self.size = size
        self.data_dir = data_dir
        self.image_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        self.eval = False
        print('LFW dataset in train mode.')
        anchor_positive_pairs_dir = self.data_dir + '/anchor_positive_pairs.txt'
        people_dir = self.data_dir + '/people.txt'

        with open(anchor_positive_pairs_dir, 'r') as text:
            self.anchor_positive_pairs_dir = text.readlines()

        with open(people_dir, 'r') as text:
            self.people_dir = text.readlines()

    def __len__(self):

        return len(self.anchor_positive_pairs_dir)

    def __getitem__(self, idx):

        anchor_name, anchor_idx, positive_idx = [x for x in self.anchor_positive_pairs_dir[idx].strip().split() if x]
        negative_name, negative_idx = self.pickAnchorNegative(anchor_name) #Randomly pick a negative sample, avoiding picking the same individual as anchor

        anchor_idx = int(anchor_idx)
        positive_idx = int(positive_idx)
        negative_idx = int(negative_idx)

        idxs = [(anchor_name, anchor_idx), (anchor_name, positive_idx), (negative_name, negative_idx)]
        images = []

        for name, idx in idxs:
            image_dir = self.getFileName(name, idx)
            image = Image.open(join(self.data_dir, image_dir))
            image = self.image_transform(image)
            images.append(image)

        return images #Array of [anchor_image, positive_image, negative_image]

    def eval(self):
        self.eval = True
        print('LFW dataset in eval mode.')

    def pickAnchorNegative(self, anchor_name):
        random.seed(time.time())
        while True:
            negative_idx = random.randrange(len(self.people_dir))
            negative_name, negative_idx = [x for x in self.people_dir[negative_idx].strip().split() if x]
            if negative_name != anchor_name:
                break

        return negative_name, negative_idx

    def sampleIndexFrom(self, max_idx):
        random.seed(time.time())
        return random.randrange(1, max_idx + 1)

    def getFileName(self, name, idx):
        return f'{name}/{name}_{idx:04d}.jpg'

