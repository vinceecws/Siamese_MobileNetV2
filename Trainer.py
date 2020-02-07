import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from os.path import join
from Siamese_MobileNetV2 import Siamese_MobileNetV2_Triplet, TripletLoss
from Faces import Faces

lr = 0.05
momentum = 0.9

class Trainer(nn.Module):
    def __init__(self, device, pretrained=False, alpha=0.2):
        super(Trainer, self).__init__()

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

