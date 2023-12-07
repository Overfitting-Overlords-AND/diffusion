import torch
from utilities import getDevice, load_latest_checkpoint
from model import DDPM, ContextUnet
import torchvision
from torchvision.utils import save_image, make_grid
import constants

def draw_image(text):
    # hardcoding these here
    device = getDevice()

    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=constants.NUM_DIMENSIONS, n_classes=constants.NUM_CLASSES, image_size=32), betas=(1e-4, 0.02), n_T=constants.NUM_TIMESTEPS, device=device, drop_prob=0.1)
    ddpm.to(device)
    load_latest_checkpoint(ddpm)

    ddpm.eval()

    with torch.no_grad():
        x_gen, _ = ddpm.single_sample(constants.IMAGE_CLASSES.index(text), (3, 32, 32), device, guide_w=constants.WEIGHT)

    grid = make_grid(x_gen*-1 + 1, nrow=10)
    save_image(grid, constants.SAVE_DIR + f"image_w{constants.WEIGHT}.png")
    print('saved image at ' + constants.SAVE_DIR + f"image_w{constants.WEIGHT}.png")
    return grid


if __name__ == "__main__":
    draw_image(8)


