import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from os.path import join
from ..models.Siamese_MobileNetV2 import Siamese_MobileNetV2_Triplet, TripletLoss

lr = 0.005
momentum = 0.9

class Trainer_TripletLoss(nn.Module):
    def __init__(self, device, pretrained=False, alpha=0.2):
        super(Trainer_TripletLoss, self).__init__()

        self.model = Siamese_MobileNetV2_Triplet(pretrained=pretrained).to(device)
        self.optim = optim.SGD([
                        {'params': self.model.parameters()},
                     ], lr=lr, momentum=momentum)

        self.loss = TripletLoss(alpha=alpha)
        self.metrics = {}

    def update(self, anchor, positive, negative):

        anchor, positive, negative = self.model(anchor, positive, negative)
        loss = self.loss(anchor, positive, negative)

        self.model.zero_grad()
        loss.backward()
        self.optim.step()

        metrics = {
            "loss/total_loss": loss.item(),
        }
        self.metrics = metrics

        return loss, anchor, positive, negative

    def get_metrics(self):
        return self.metrics

    def load(self, checkpoint):
        self.model.load_state_dict(checkpoint['weight'])

    def save(self, save_dir, iterations):
        weight_fn = join(save_dir, "siamese_mobilenet_v2_pretrained_triplet%d.pkl" % iterations)

        state = {
            'weight': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'iterations': iterations,
        }

        torch.save(state, weight_fn)

class Trainer_TripletLoss_NegativeHard(nn.Module):
    def __init__(self, device, mini_batch_size=4, pretrained=False, alpha=0.2):
        super(Trainer_TripletLoss_NegativeHard, self).__init__()

        self.model = Siamese_MobileNetV2_Triplet(pretrained=pretrained).to(device)
        self.optim = optim.SGD([
                        {'params': self.model.parameters()},
                     ], lr=lr, momentum=momentum)

        self.loss = TripletLoss(alpha=alpha)
        self.metrics = {}
        self.mini_batch_size = mini_batch_size
        self.num_iterations = None

    def setNumIterations(self, num_samples):
        self.num_iterations = num_samples / self.mini_batch_size

    def update(self):
        self.optim.step()
        self.optim.zero_grad()
        self.num_iterations = None #Reset for new batch

    def accumulate(self, anchor, positive, negative): #Call model.train() first
        assert self.num_iterations is not None, 'num_iterations must be set before every new training batch, call setNumIterations() first.'
        assert not self.model.inEval(), 'Model must be in train mode. Call model.train() first.'
        anchor, positive, negative = self.model(anchor, positive, negative)
        loss = self.loss(anchor, positive, negative) / self.num_iterations

        metrics = {
            "loss": loss.item(),
        }

        self.metrics = metrics

        loss.backward()

        return loss, anchor, positive, negative

    def forward_prop(self, anchor, positive, negative): #Call model.train() first
        assert not self.model.inEval(), 'Model must be in train mode. Call model.train() first.'
        anchor, positive, negative = self.model(anchor, positive, negative)
        self.optim.zero_grad() #Clear gradients

        return anchor, positive, negative

    def get_metrics(self):
        return self.metrics

    def load(self, checkpoint):
        self.model.load_state_dict(checkpoint['weight'])

    def save(self, save_dir, iterations):
        weight_fn = join(save_dir, "siamese_mobilenet_v2_pretrained_triplet_negative_hard%d.pkl" % iterations)

        state = {
            'weight': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'iterations': iterations,
        }

        torch.save(state, weight_fn)