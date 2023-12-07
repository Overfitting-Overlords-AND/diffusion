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
from utilities import getDevice 
import wandbWrapper as wandb
import constants

def train_mnist():

    # hardcoding these here
    n_epoch = constants.NUM_OF_EPOCHS
    batch_size = constants.BATCH_SIZE
    n_T = constants.NUM_TIMESTEPS
    device = getDevice()
    n_classes = constants.NUM_CLASSES
    n_feat = constants.NUM_DIMENSIONS
    lrate = constants.LEARNING_RATE
    save_dir = './data/diffusion_outputs_CIFAR10/'

    wandb.init()
    
    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes, image_size=32), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    # optionally load a model
    #ddpm.load_state_dict(torch.load("./pretrained_model/model_39.pth", map_location=torch.device('cpu')))

    tf = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1

    # Load the training data
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=tf)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size,
                                            shuffle=True, num_workers=2)


    # dataset = MNIST("./data", train=True, download=True, transform=tf)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

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
            wandb.log({"loss_ema": loss_ema})
            optim.step()
        
        # optionally save model
        torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
        print('saved model at ' + save_dir + f"model_{ep}.pth")

    wandb.finish()


if __name__ == "__main__":
    train_mnist()
