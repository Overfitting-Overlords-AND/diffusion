import constants
import wandb

def init():
  # start a new wandb run to track this script
  if constants.WANDB_ON:
    wandb.init(
        # set the wandb project where this run will be logged
        project="Diffusion",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": constants.LEARNING_RATE,
        "batch_size": constants.BATCH_SIZE,
        "num_epochs": constants.NUM_OF_EPOCHS,
        "num_classes": constants.NUM_CLASSES,
        "margin" : constants.NUM_WORKERS,
        "num_timesteps" : constants.NUM_TIMESTEPS,
        "num_dimensions" : constants.NUM_DIMENSIONS
        }
    )

def log(metrics):
  if constants.WANDB_ON:
    wandb.log(metrics)

def finish():
  if constants.WANDB_ON:
    wandb.finish()