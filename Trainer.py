import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from os.path import join
from Siamese_MobileNetV2 import Siamese_MobileNetV2, ContrastiveLoss
from Faces import Faces

lr = 0.001
momentum = 0.9

class Trainer(nn.Module):
    def __init__(self, device, pretrained=False, margin=1.0):
        super(Trainer, self).__init__()

        self.model = Siamese_MobileNetV2(pretrained=pretrained).to(device)
        self.optim = optim.SGD([
                        {'params': self.model.parameters()},
                     ], lr=lr, momentum=momentum)

        self.loss = ContrastiveLoss(margin=margin)
        self.metrics = {}

    def update(self, input1, label1, input2, label2):
        loss = 0.0

        output1, output2 = self.model(input1, input2)
        label = label1 != label2 #0 if label1 == label2, else 1
        loss = self.loss(output1, output2, label.int())

        self.model.zero_grad()
        loss.backward()
        self.optim.step()

        metrics = {
            "loss/total_loss": loss.item(),
        }
        self.metrics = metrics

        return output1, output2, loss

    def get_metrics(self):
        return self.metrics

    def load(self, checkpoint):
        self.model.load_state_dict(checkpoint['weight'])

    def save(self, save_dir, iterations):
        weight_fn = join(save_dir, "siamese_mobilenet_v2_pretrained_%d.pkl" % iterations)

        state = {
            'weight': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'iterations': iterations,
        }

        torch.save(state, weight_fn)

