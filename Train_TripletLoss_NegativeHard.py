import os
import argparse
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from tensorboardX import SummaryWriter

from src.utils.LFW import LFW_Triple_Negative_Hard_Mining
from src.utils.Trainer import Trainer_TripletLoss_NegativeHard

batch_size = 380
mini_batch_size = 4
num_epochs = 30
alpha = 0.2
data_dir = "./lfw"
weight_dir = "./weights/augmented"
embeddings_dir = "./embeddings"
size = (224, 224) #input and output size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainset = LFW_Triple_Negative_Hard_Mining(data_dir, embeddings_dir, size, mini_batch_size=mini_batch_size)
trainer = Trainer_TripletLoss_NegativeHard(device, pretrained=True, alpha=alpha)
trainer.model.train()
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
        it = 0
    epochs = num_epochs
    batches = trainset.batches()
    for e in range(1, epochs + 1):
            for b in range(batches):
                trainset.setBatchInd(b) #Set batch index
                print(f'Epoch {e}/{epochs}, Batch {b + 1}/{batches}:')
                #Forward through mini batch, select triplets and build dataset
                trainset.normal() #Normal pick without using triplet loss constraint
                dataloader = torch.utils.data.DataLoader(
                                trainset, batch_size=mini_batch_size,
                                shuffle=False, num_workers=0)
                print('Creating embeddings for negative hard mining...')
                t = tqdm(dataloader)
                embeddings = []
                for i, data in enumerate(t):
                    # Input
                    anchor = data[0].to(device)
                    positive = data[1].to(device)
                    negative = data[2].to(device)

                    # Get embeddings
                    output = trainer.forward_prop(anchor, positive, negative)
                    torch.save(output, f'./embeddings/emb_{i:03d}.pt')

                # Build hard samples
                print('Building semi-hard negative triplets...')
                trainset.buildHardNegativeSamples(b, alpha)    
                
                #Train with selected triplets that violate triplet loss constraint
                trainset.hard() #Hard pick using triplet loss constraint
                num_samples = len(trainset)
                trainer.setNumIterations(num_samples)
                dataloader = torch.utils.data.DataLoader(
                                trainset, batch_size=mini_batch_size,
                                shuffle=False, num_workers=0, drop_last=True)
                print(f'Training with selected triplets... {num_samples} triplets')
                sum_loss = 0
                t = tqdm(dataloader)
                for i, data in enumerate(t):
                    # Input
                    anchor = data[0].to(device)
                    positive = data[1].to(device)
                    negative = data[2].to(device)

                    # Train
                    output = trainer.accumulate(anchor, positive, negative)

                    # Log
                    metrics = trainer.get_metrics()
                    for k, v in metrics.items():
                        writer.add_scalar(k, v, it)

                    t.set_postfix(loss=metrics['loss'])
                    sum_loss += metrics['loss']
                    it += 1

                # Update after every mini-batch
                print('Updating weights...')
                trainer.update()

                # Save after every mini-batch
                print('Saving checkpoint...')
                trainer.save(weight_dir, it)

            print('Average loss @ epoch: {}'.format((sum_loss / i)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", help="Load weight from file")

    args = parser.parse_args()

    main(args)

