from enum import Enum

LEARNING_RATE = 1e-4
BATCH_SIZE = 128

NUM_OF_EPOCHS = 100
NUM_WORKERS = 6

WANDB_ON = True
NUM_CLASSES = 10
NUM_TIMESTEPS = 400
NUM_DIMENSIONS = 128
DROP_PROB = 0.1
BETAS = (1e-4, 0.02)

WEIGHT = 2

SAVE_DIR = './data/diffusion_outputs_CIFAR10/'

IMAGE_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

class Mode(Enum):
    CIFAR = 1
    MNIST = 2

MODE = Mode.CIFAR

#CIFAR cnostants
CIFAR_IMAGE_SIZE = 32
CIFAR_IMAGE_DEPTH = 3