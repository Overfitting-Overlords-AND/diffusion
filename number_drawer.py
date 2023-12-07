import torch
from utilities import getDevice 
from model import DDPM, ContextUnet
import torchvision
from torchvision.utils import save_image, make_grid

def draw_number(text):
    # hardcoding these here
    n_T = 400 # 500
    device = getDevice()
    n_classes = 10


    n_feat = 128 # 128 ok, 256 better (but slower)
    save_dir = './data/diffusion_outputs10/'
    # ws_test = [0.0, 0.5, 2.0] # strength of generative guidance
    w = 2.0

    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)
    ddpm.load_state_dict(torch.load("./pretrained_model/model_39.pth", map_location=torch.device('cpu')))

    # dataset = MNIST("./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    ddpm.eval()

    with torch.no_grad():
        x_gen, _ = ddpm.single_sample(text, (1, 28, 28), device, guide_w=w)

    grid = make_grid(x_gen*-1 + 1, nrow=10)
    save_image(grid, save_dir + f"image_w{w}.png")
    print('saved image at ' + save_dir + f"image_w{w}.png")
    # img = torchvision.transforms.functional.to_pil_image(x_gen.squeeze())
    # return img


if __name__ == "__main__":
    # draw_number(8).show()
    draw_number(8)


