from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
from model import DDPM, ContextUnet
from utilities import getDevice, save_checkpoint, load_latest_checkpoint, find_latest_epoch_file
import wandbWrapper as wandb
import constants

def train_mnist():

    # hardcoding these here
    device = getDevice()
    n_classes = constants.NUM_CLASSES
    n_feat = constants.NUM_DIMENSIONS
    lrate = constants.LEARNING_RATE

    wandb.init()
    
    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes, image_size=32), betas=(1e-4, 0.02), n_T=constants.NUM_TIMESTEPS, device=device, drop_prob=constants.DROP_PROB)
    ddpm.to(device)

    tf = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load the training data
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=tf)
    dataloader = torch.utils.data.DataLoader(dataset, constants.BATCH_SIZE,
                                            shuffle=True, num_workers=2)

    start_epoch = load_latest_checkpoint(ddpm)

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(start_epoch,constants.NUM_OF_EPOCHS):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/constants.NUM_OF_EPOCHS)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        
        wandb.log({"loss_ema": loss_ema})
        # optionally save model
        save_checkpoint(ddpm.state_dict(), f"{constants.SAVE_DIR}model_{ep}.pth")
        #torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
        print(f"saved model at {constants.SAVE_DIR}model_{ep}.pth")

    wandb.finish()


if __name__ == "__main__":
    train_mnist()

