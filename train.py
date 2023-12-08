from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
from model import DDPM, ContextUnet
from utilities import getDevice 
import constants

def train_mnist():

    # hardcoding these here
    device = getDevice()
        
    ddpm = DDPM(nn_model=ContextUnet(in_channels=constants.MNIST_IMAGE_DEPTH, n_feat=constants.MNIST_NUM_DIMENSIONS, n_classes=constants.NUM_CLASSES), betas=constants.BETAS, n_T=constants.NUM_TIMESTEPS, device=device, drop_prob=constants.DROP_PROB)
    ddpm.to(device)

    # optionally load a model
    ddpm.load_state_dict(torch.load("./pretrained_model/model_39.pth", map_location=torch.device('cpu')))

    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1

    dataset = MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=constants.MNIST_BATCH_SIZE, shuffle=True, num_workers=constants.NUM_WORKERS)
    optim = torch.optim.Adam(ddpm.parameters(), lr=constants.LEARNING_RATE)

    for ep in range(constants.NUM_OF_EPOCHS):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = constants.LEARNING_RATE*(1-ep/constants.NUM_OF_EPOCHS)

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
        
        # optionally save model
        torch.save(ddpm.state_dict(), constants.MNIST_SAVE_DIR + f"model_{ep}.pth")
        print('saved model at ' + constants.MNIST_SAVE_DIR + f"model_{ep}.pth")

if __name__ == "__main__":
    train_mnist()

