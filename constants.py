from enum import Enum

LEARNING_RATE = 1e-4
NUM_OF_EPOCHS = 100
NUM_WORKERS = 6

WANDB_ON = True
NUM_CLASSES = 10
NUM_TIMESTEPS = 400
DROP_PROB = 0.1
BETAS = (1e-4, 0.02)

WEIGHT = 2

class Mode(Enum):
    CIFAR = 1
    MNIST = 2

MODE = Mode.CIFAR

#CIFAR constants
CIFAR_IMAGE_SIZE = 32
CIFAR_IMAGE_DEPTH = 3
CIFAR_SAVE_DIR = './data/diffusion_outputs_CIFAR10/'
CIFAR_IMAGE_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
CIFAR_BATCH_SIZE = 64
CIFAR_NUM_DIMENSIONS = 256

#MNIST cnostants
MNIST_IMAGE_SIZE = 28
MNIST_IMAGE_DEPTH = 1
MNIST_SAVE_DIR = './data/diffusion_outputs10/'
MNIST_BATCH_SIZE = 128
MNIST_NUM_DIMENSIONS = 128
