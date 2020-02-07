from os.path import join
import random
import torch
import time
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import Image
from torch.utils.data import Dataset

class Faces(Dataset):
    def __init__(self, data_dir, size, pair=False, crop_face=False):
        self.size = size
        self.data_dir = data_dir
        self.pair = pair
        self.crop_face = crop_face
        self.image_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        self.eval = False
        print('Faces dataset in train mode.')
        labels_dir = self.data_dir + '/labels'
        labelsOnly_dir = self.data_dir + '/labelsOnly'
        boxes_dir = self.data_dir + '/boxes'

        with open(labels_dir, 'r') as text:
            self.labels_dir = text.readlines()

        with open(labelsOnly_dir, 'r') as text:
            self.labelsOnly_dir = np.array([int(i.rstrip()) for i in text.readlines()])

        with open(boxes_dir, 'r') as text:
            self.boxes_dir = text.readlines()

    def __len__(self):

        return len(self.labels_dir)

    def __getitem__(self, idx):

        image_dir, label = self.labels_dir[idx].rstrip().split(' ')
        image = Image.open(join(self.data_dir, image_dir))
        if self.crop_face:
            bbox = list(map(float, self.boxes_dir[idx].rstrip().split(' ')))
            image = crop(image, bbox)
        if not self.eval:
            image = image_augmentation(image)
        image = self.image_transform(image)
        label = int(label)

        if self.pair:
            pair_idx = self.rollDice(idx, label, p=0.5) #50/50 chance of either picking same label, or a different label
            image_dir2, label2 = self.labels_dir[pair_idx].rstrip().split(' ')
            image2 = Image.open(join(self.data_dir, image_dir2))
            if self.crop_face:
                bbox2 = list(map(float, self.boxes_dir[pair_idx].rstrip().split(' ')))
                image2 = crop(image2, bbox2)
            if not self.eval:
                image2 = image_augmentation(image2)
            image2 = self.image_transform(image2)
            label2 = int(label2)

            return image, label, image2, label2

        return image, label

    def eval(self):
        self.eval = True
        print('Faces dataset in eval mode.')

    def rollDice(self, idx, label, p=0.5): #only works if shuffle=False
        assert self.labelsOnly_dir[idx] == label
        random.seed(time.time())
        outcome = random.uniform(0.0, 1.0)
        if outcome >= p: #Pick same label
            choices = np.where(self.labelsOnly_dir == label)[0]
            if choices.size > 2:
                choices = np.delete(choices, np.where(choices == idx))
            else:
                choices = np.argwhere(self.labelsOnly_dir != label) #Pick different label if not enough same labels
        else: #Pick different label
            choices = np.argwhere(self.labelsOnly_dir != label)

        pair_idx = np.random.choice(choices.flatten())

        return pair_idx

def image_augmentation(image):
    random_gamma = random.uniform(0.8, 1.2)
    random_brightness = random.uniform(0.5, 2.0)
    random_flip = random.uniform(0.0, 1.0)

    TF.adjust_gamma(image, random_gamma)
    TF.adjust_brightness(image, random_brightness)

    if random_flip > 0.5:
        image = TF.hflip(image)

    return image

def crop(image, bbox, squareify=True):
    x_bot_left, y_bot_left, x_top_left, y_top_left, x_top_right, y_top_right, x_bot_right, y_bot_right = bbox
    left = int(min(x_bot_left, x_top_left))
    top = int(min(y_top_left, y_top_right))
    right = int(max(x_top_right, x_bot_right))
    bot = int(max(y_bot_left, y_bot_right))
    if squareify:
        new_size = max(right - left, bot - top)
        if new_size < 0:
            print('NEGATIVE')
        right = left + new_size
        bot = top + new_size

    return image.crop((left, top, right, bot)) #CROP IMAGE

