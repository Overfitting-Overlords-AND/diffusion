import torch
from utilities import getDevice 
from model import DDPM, ContextUnet
from torchvision.utils import save_image, make_grid
import constants

def draw_number(text):
    
    device = getDevice()

    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=constants.NUM_DIMENSIONS, n_classes=constants.NUM_CLASSES), betas=constants.BETAS, n_T=constants.NUM_TIMESTEPS, device=device, drop_prob=0.1)
    ddpm.to(device)
    ddpm.load_state_dict(torch.load("./pretrained_model/model_39.pth", map_location=torch.device('cpu')))

    ddpm.eval()

    with torch.no_grad():
        x_gen, _ = ddpm.single_sample(constants.MNIST_IMAGE_DEPTH, (1, constants.MNIST_IMAGE_SIZE, constants.MNIST_IMAGE_SIZE), device, guide_w=constants.WEIGHT)

    grid = make_grid(x_gen*-1 + 1, nrow=10)
    save_image(grid, constants.MNIST_SAVE_DIR + f"image_w{constants.WEIGHT}.png")
    print('saved image at ' + constants.MNIST_SAVE_DIR + f"image_w{constants.WEIGHT}.png")
    return grid


if __name__ == "__main__":
    draw_number(8)

