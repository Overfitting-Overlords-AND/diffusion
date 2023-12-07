import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
from model import DDPM, ContextUnet
from utilities import getDevice, load_latest_checkpoint
import constants

def eval_cifar10():
    # hardcoding these here
    n_T = constants.NUM_TIMESTEPS # 500
    device = getDevice()
    n_classes = constants.NUM_CLASSES
    batch_size = constants.BATCH_SIZE

    n_feat = constants.NUM_DIMENSIONS
    ws_test = [0.0, 0.5, 2.0] # strength of generative guidance

    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes, image_size=32), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)
    load_latest_checkpoint(ddpm)

    transforms.Compose([transforms.ToTensor()])

    # Load the training data
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    dataloader = torch.utils.data.DataLoader(dataset, constants.BATCH_SIZE, shuffle=True, num_workers=constants.NUM_WORKERS)
    ddpm.eval()

    for x, c in dataloader:
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        with torch.no_grad():
            n_sample = 4*n_classes
            for _, w in enumerate(ws_test):
                x_gen, _ = ddpm.sample(n_sample, (3, 32, 32), device, guide_w=w)

                # append some real images at bottom, order by class also
                x_real = torch.Tensor(x_gen.shape).to(device)
                for k in range(n_classes):
                    for j in range(int(n_sample/n_classes)):
                        try: 
                            idx = torch.squeeze((c == k).nonzero())[j]
                        except:
                            idx = 0
                        x_real[k+(j*n_classes)] = x[idx]

                x_all = torch.cat([x_gen, x_real])
                grid = make_grid(x_all*-1 + 1, nrow=10)
                save_image(grid, constants.SAVE_DIR + f"image_w{w}.png")
                print('saved image at ' + constants.SAVE_DIR + f"image_w{w}.png")

                # if ep%5==0 or ep == int(n_epoch-1):
                #     # create gif of images evolving over time, based on x_gen_store
                #     fig, axs = plt.subplots(nrows=int(n_sample/n_classes), ncols=n_classes,sharex=True,sharey=True,figsize=(8,3))
                #     def animate_diff(i, x_gen_store):
                #         print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                #         plots = []
                #         for row in range(int(n_sample/n_classes)):
                #             for col in range(n_classes):
                #                 axs[row, col].clear()
                #                 axs[row, col].set_xticks([])
                #                 axs[row, col].set_yticks([])
                #                 # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                #                 plots.append(axs[row, col].imshow(-x_gen_store[i,(row*n_classes)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                #         return plots
                #     ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
                #     ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
                #     print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")

if __name__ == "__main__":
    eval_cifar10()

