import torch
from utilities import getDevice, load_latest_checkpoint
from model import DDPM, ContextUnet
from torchvision.utils import save_image, make_grid
import constants
import matplotlib.pyplot as plt

def draw_image(text):
    # hardcoding these here
    device = getDevice()

    ddpm = DDPM(nn_model=ContextUnet(in_channels=constants.CIFAR_IMAGE_DEPTH, n_feat=constants.CIFAR_NUM_DIMENSIONS, n_classes=constants.NUM_CLASSES, image_size=constants.CIFAR_IMAGE_SIZE), betas=constants.BETAS, n_T=constants.NUM_TIMESTEPS, device=device, drop_prob=constants.DROP_PROB)
    ddpm.to(device)
    load_latest_checkpoint(ddpm)

    ddpm.eval()

    with torch.no_grad():
        x_gen, _ = ddpm.single_sample(constants.CIFAR_IMAGE_CLASSES.index(text), (constants.CIFAR_IMAGE_DEPTH, constants.CIFAR_IMAGE_SIZE, constants.CIFAR_IMAGE_SIZE), device, guide_w=constants.WEIGHT)

    # Display the image
    grid = make_grid(x_gen, nrow=10)
    save_image(grid, constants.CIFAR_SAVE_DIR + f"image_w{constants.WEIGHT}.png")
    print('saved image at ' + constants.CIFAR_SAVE_DIR + f"image_w{constants.WEIGHT}.png")
    return grid


if __name__ == "__main__":
    draw_image("horse")


