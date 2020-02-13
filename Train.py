import os
import argparse
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils.LFW import LFW
from Trainer import Trainer_TripletLoss

batch_size = 8
num_epochs = 30

data_dir = "./lfw"
weight_dir = "./weights"
size = (224, 224) #input and output size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainset = LFW(data_dir, size)
dataloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size,
                shuffle=False, num_workers=0)

trainer = Trainer_TripletLoss(device, pretrained=True)
writer = SummaryWriter()

def main(args):
    if args.resume:
        assert os.path.isfile(args.resume), \
            "%s is not a file" % args.resume

        checkpoint = torch.load(args.resume)
        trainer.load(checkpoint)
        it = checkpoint["iterations"] + 1

        print("Loaded checkpoint '{}' (iterations {})"
            .format(args.resume, checkpoint['iterations']))
    else:
        it = 1

    for e in range(1, num_epochs + 1):
        print("Epoch %d" % e)

        sum_loss = 0
        t = tqdm(dataloader)
        for i, data in enumerate(t):
            # Input
            anchor = data[0].to(device)
            positive = data[1].to(device)
            negative = data[2].to(device)

            # Train
            output = trainer.update(anchor, positive, negative)

            # Log
            metrics = trainer.get_metrics()
            for k, v in metrics.items():
                writer.add_scalar(k, v, it)

            t.set_postfix(loss=metrics['loss/total_loss'])

            sum_loss += metrics['loss/total_loss']

            # Save
            if it % 100 == 0:
                print('Saving checkpoint...')
                trainer.save(weight_dir, it)


            it += 1

        print('Average loss @ epoch: {}'.format((sum_loss / (i * batch_size))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", help="Load weight from file")

    args = parser.parse_args()

    main(args)