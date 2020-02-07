import os
import argparse
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from tensorboardX import SummaryWriter

from Faces import Faces
from Trainer import Trainer

batch_size = 4
num_epochs = 5

data_dir = "./Faces_cropped"
weight_dir = "./weights"
size = (224, 224) #input and output size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainset = Faces(data_dir, size, pair=True, crop_face=False)
dataloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size,
                shuffle=False, num_workers=0)

trainer = Trainer(device, pretrained=True)
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
            image1 = data[0].to(device)
            label1 = data[1]
            image2 = data[2].to(device)
            label2 = data[3]

            # Train
            output = trainer.update(image1, label1, image2, label2)

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

