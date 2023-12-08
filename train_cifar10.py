from tqdm import tqdm
import torch
import torchvision
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from model import DDPM, ContextUnet
from utilities import getDevice, save_checkpoint, load_latest_checkpoint, find_latest_epoch_file
import wandbWrapper as wandb
import constants

def train_cifar10():

    # hardcoding these here
    device = getDevice()
    wandb.init()
    
    ddpm = DDPM(nn_model=ContextUnet(in_channels=constants.CIFAR_IMAGE_DEPTH, n_feat=constants.NUM_DIMENSIONS, n_classes=constants.NUM_CLASSES, image_size=constants.CIFAR_IMAGE_SIZE), betas=constants.BETAS, n_T=constants.NUM_TIMESTEPS, device=device, drop_prob=constants.DROP_PROB)
    ddpm.to(device)

    tf = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load the training data
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=tf)
    dataloader = torch.utils.data.DataLoader(dataset, constants.BATCH_SIZE,
                                            shuffle=True, num_workers=constants.NUM_WORKERS)

    start_epoch = load_latest_checkpoint(ddpm)

    optim = torch.optim.Adam(ddpm.parameters(), lr=constants.LEARNING_RATE)

    for ep in range(start_epoch,constants.NUM_OF_EPOCHS):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = constants.LEARNING_RATE*(1-ep/(start_epoch+constants.NUM_OF_EPOCHS))

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
        save_checkpoint(ddpm.state_dict(), f"{constants.CIFAR_SAVE_DIR}model_{ep}.pth")
        print(f"saved model at {constants.CIFAR_SAVE_DIR}model_{ep}.pth")

    wandb.finish()


if __name__ == "__main__":
    train_cifar10()

