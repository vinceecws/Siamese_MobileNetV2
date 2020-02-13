from os import listdir
from os.path import join
import random
import torch
import time
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from ..models.Siamese_MobileNetV2 import TripletLoss as tploss
from PIL import Image
from torch.utils.data import Dataset

class LFW(Dataset):
    def __init__(self, data_dir, size):
        super(LFW, self).__init__()
        self.size = size
        self.data_dir = data_dir
        self.image_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        self.eval_mode = False
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

        anchor_name, anchor_idx, anchor_name, positive_idx, negative_name, negative_idx = self.randomPick(idx)

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
        self.eval_mode = True
        print('LFW dataset in eval mode.')

    def pickAnchorNegative(self, anchor_name):
        random.seed(time.time())
        while True:
            negative_idx = random.randrange(len(self.people_dir))
            negative_name, negative_idx = [x for x in self.people_dir[negative_idx].strip().split() if x]
            negative_idx = random.randrange(1, int(negative_idx) + 1)
            if negative_name != anchor_name:
                break

        return negative_name, negative_idx


    def sampleIndexFrom(self, max_idx):
        random.seed(time.time())
        return random.randrange(1, max_idx + 1)

    def getFileName(self, name, idx):
        return f'{name}/{name}_{idx:04d}.jpg'

    def randomPick(self, idx):
        anchor_name, anchor_idx, positive_idx = [x for x in self.anchor_positive_pairs_dir[idx].strip().split() if x]
        negative_name, negative_idx = self.pickAnchorNegative(anchor_name) #Randomly pick a negative sample, avoiding picking the same individual as anchor

        return anchor_name, anchor_idx, anchor_name, positive_idx, negative_name, negative_idx

'''
    Dataset class for Labeled Faces in the Wild.
    Note: If used with PyTorch Dataloader, shuffle=False
'''

class LFW_Triple_Negative_Hard_Mining(Dataset):
    def __init__(self, data_dir, embeddings_dir, size, threshold=20, mini_batch_size=4, batch_size=380, shuffle=False): 
    #62 people have 20 images & above, total 62 * (20 * 19) = 23560 triplets
    #23560 / 380 = 62 iterations of 380 samples in each mini-batch 
    #(each mini-batch has one main identity, of which all A-P pairs are selected from, plus random A-N pairs)
        super(LFW_Triple_Negative_Hard_Mining, self).__init__()
        self.size = size
        self.data_dir = data_dir
        self.embeddings_dir = embeddings_dir
        self.image_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
            ])
        self.eval_mode = False
        self.shuffle = shuffle
        self.threshold = threshold
        self.mini_batch_size = mini_batch_size
        self.batch_size = batch_size
        self.batch_ind = 0
        print('LFW dataset in train mode.')
        anchor_positive_pairs_dir = self.data_dir + '/anchor_positive_pairs.txt'
        people_dir = self.data_dir + '/people.txt'

        with open(anchor_positive_pairs_dir, 'r') as text:
            self.anchor_positive_pairs_dir = text.readlines()

        with open(people_dir, 'r') as text:
            self.people_dir = text.readlines()

        self.people_above_threshold = self.getAboveThreshold()
        self.people_above_threshold = self.expandAndSplit(self.people_above_threshold, self.threshold, self.batch_size, shuffle=self.shuffle)
        self.hard_negative_triplets = None

    def batches(self):
        return np.shape(self.people_above_threshold)[0]

    def __len__(self):
        assert not self.hard_mode or (self.hard_mode and self.hard_negative_triplets), 'In hard mode, buildHardNegativeSamples() must be called before use.'

        return np.shape(self.hard_negative_triplets)[0] if self.hard_mode else np.shape(self.people_above_threshold[self.batch_ind])[0]

    def __getitem__(self, idx):

        if self.hard_mode:
            anchor_name, anchor_idx, positive_name, positive_idx, negative_name, negative_idx = self.hardNegativePick(idx)
        else:
            anchor_name, anchor_idx, positive_name, positive_idx, negative_name, negative_idx = self.normalPick(self.batch_ind, idx)

        anchor_idx = int(anchor_idx)
        positive_idx = int(positive_idx)
        negative_idx = int(negative_idx)

        idxs = [(anchor_name, anchor_idx), (positive_name, positive_idx), (negative_name, negative_idx)]
        images = []

        for name, idx in idxs:
            image_dir = self.getFileName(name, idx)
            image = Image.open(join(self.data_dir, image_dir))
            if not self.eval_mode and self.hard_mode:
                image = image_augmentation(image)
            image = self.image_transform(image)
            images.append(image)

        return images #List of [anchor_image, positive_image, negative_image]

    def eval(self):
        self.eval_mode = True
        print('LFW dataset in eval mode.')

    def normal(self):
        self.hard_mode = False
        print('LFW dataset in normal mode.')

    def hard(self):
        assert self.hard_negative_triplets is not None, 'Call buildHardNegativeSamples() with embeddings before enabling hard mode.'
        self.hard_mode = True
        print('LFW dataset in hard mode.')

    def setBatchInd(self, batch_ind):
        assert batch_ind in range(self.batches())
        self.batch_ind = batch_ind

    def pickAnchorNegative(self, anchor_name):
        random.seed(time.time())
        while True:
            negative_idx = random.randrange(len(self.people_dir))
            negative_name, negative_idx = [x for x in self.people_dir[negative_idx].strip().split() if x]
            negative_idx = random.randrange(1, int(negative_idx) + 1)
            if negative_name != anchor_name:
                break

        return negative_name, negative_idx

    def buildHardNegativeSamples(self, batch, alpha): #Embeddings in (N x 3 x 128)
        indices = []
        sorted_dir = [f for f in listdir(self.embeddings_dir) if not f.startswith('.')] #Ignore hidden files
        sorted_dir.sort()
        for mini_batch, embedding in enumerate(sorted_dir):
            emb_dir = join(self.embeddings_dir, embedding)
            anchor_emb, positive_emb, negative_emb = torch.load(emb_dir)
            pos_dist_sqr = tploss.dist(anchor_emb, positive_emb)
            neg_dists_sqr = tploss.dist(anchor_emb, negative_emb)
            #ind = torch.where(((neg_dists_sqr - pos_dist_sqr) < alpha) & (pos_dist_sqr < neg_dists_sqr))[0] #Maximal hard with (A-N - A-P < alpha), semi-hard with additional (A-P < A-N)
            ind = torch.where(((neg_dists_sqr - pos_dist_sqr) < alpha))[0]
            ind += mini_batch * self.mini_batch_size
            ind = [int(x) for x in ind]
            indices.extend(ind)

        self.hard_negative_triplets = [self.people_above_threshold[batch][i] for i in indices]

    def sampleIndexFrom(self, max_idx):
        random.seed(time.time())
        return random.randrange(1, max_idx + 1)

    def getFileName(self, name, idx):
        return f'{name}/{name}_{idx:04d}.jpg'

    def normalPick(self, batch, idx):

        return self.people_above_threshold[batch][idx]

    def hardNegativePick(self, idx):
        assert self.hard_negative_triplets is not None

        return self.hard_negative_triplets[idx]

    def getAboveThreshold(self):
        count = 0
        people = []
        for line in self.people_dir:
                name, num = [x for x in line.strip().split() if x]
                if int(num) >= self.threshold:
                        count +=1
                        people.append((name, num))

        return people

    def expandAndSplit(self, people, threshold, batch_size, shuffle=False):
        num_samples = len(people) * threshold * (threshold - 1)
        assert num_samples % batch_size == 0, f'With {len(people)} identities at {threshold} images/identity, samples of length \
            {len(people)} * {threshold} * ({threshold} - 1) = {num_samples} is not divisible by batch_size of {batch_size}'

        if shuffle:
            random.seed(time.time())
            random.shuffle(people)

        batched_people = []

        for anchor_name, num in people:
            batch = []
            num = int(num)
            assert num >= threshold
            indices_a = random.sample(range(1, num + 1), threshold)
            for anchor_idx in indices_a:
                indices_p = random.sample([x for x in range(1, num + 1) if x != anchor_idx], threshold - 1)
                for positive_idx in indices_p:
                    negative_name, negative_idx = self.pickAnchorNegative(anchor_name)
                    batch.append((anchor_name, anchor_idx, anchor_name, positive_idx, negative_name, negative_idx))

            batched_people.append(batch)

        return batched_people

def image_augmentation(image):
    random_gamma = random.uniform(0.8, 1.2)
    random_brightness = random.uniform(0.5, 2.0)
    random_flip = random.uniform(0.0, 1.0)

    TF.adjust_gamma(image, random_gamma)
    TF.adjust_brightness(image, random_brightness)
    if random_flip > 0.5:
        image = TF.hflip(image)

    return image

