from enum import Enum

LEARNING_RATE = 1e-4
BATCH_SIZE = 256

NUM_OF_EPOCHS = 100
NUM_WORKERS = 5

WANDB_ON = True
NUM_CLASSES = 10
NUM_TIMESTEPS = 400
NUM_DIMENSIONS = 256
DROP_PROB = 0.1
WEIGHT = 2

SAVE_DIR = './data/diffusion_outputs_CIFAR10/'

IMAGE_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

class Mode(Enum):
    CIFAR = 1
    MNIST = 2

MODE = Mode.CIFAR
