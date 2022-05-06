"""
This is a boilerplate pipeline 'train_model_pipeline'
generated using Kedro 0.17.7
"""

import torch
import os

from kedro.io import DataCatalog, MemoryDataSet
from pytorch_lightning.loggers import WandbLogger
from Project_DL.LightningModels.model import ErCaNet
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin

from Project_DL.DataClasses.dataloaders import DataModuleClass


data = {}
memory_dataset = MemoryDataSet(data)
data_catalog = DataCatalog({"dataset": memory_dataset})
# max_batches = 3
# data_path = 'data/all_unpickle'
# font_path = 'data/fonts'
# model_save_path = 'models/'
# true_randomness = False
# resize_up_to = 256
# batch_size = 48
# loader_workers = 8

# data
def load_dataset(dataset_params):
  """Function that loads all datasets - training, testing and validation

  Args:
      dataset_params (dict): A dictionary with parameters to build dataset class.

  Returns:
      DataLoader, DataLoader, DataLoader: Dataloaders for train, test and validation datasets
  """
  dataset = DataModuleClass(
    dataset_params["max_batches"], 
    dataset_params["data_path"], 
    dataset_params["font_path"], 
    dataset_params["resize_up_to"], 
    dataset_params["true_randomness"]
  )
  
  dataset.setup()
  
  train_loader = dataset.train_dataloader(
    dataset_params["batch_size"],
    shuffle = True, 
    num_workers=dataset_params["loader_workers"]
  )
  
  test_loader = dataset.test_dataloader(
    dataset_params["batch_size"],
    shuffle = False,
    num_workers=dataset_params["loader_workers"]
  )  
  
  val_loader = dataset.val_dataloader(
    dataset_params["batch_size"],
    shuffle = False,
    num_workers=dataset_params["loader_workers"]
  )
  return train_loader, test_loader, val_loader

# model
def get_model(logger):
  """Function that creates the caption-erasing model

  Args:
      logger (WandbLogger): Logger for logging training progress

  Returns:
      ErCaNet: Model that erases captions from images
  """
  model = ErCaNet(logger.name)
  return model

def get_logger(model_name):
  """Function that creates a logger for logging training progress

  Args:
      model_name (str): A name of current model to display in wandb.

  Returns:
      WandbLogger: Logger that logs information about the training progress
  """
  wandb_logger = WandbLogger(name=model_name, project='ErCaNet')
  return wandb_logger

# trainer
def get_trainer(wandb_logger, trainer_params):
  """Function that creates a trainer that manages the training process of the model

  Args:
      wandb_logger (WandbLogger): Logger that logs information about the training progress
      trainer_params (dict): A dictionary with parameters to build trainer class.

  Returns:
      pytorch_lightning.Trainer: Trainer that manages the training process of the model
  """
  trainer = Trainer(
    accelerator=trainer_params["accelerator"],
    gpus=trainer_params["gpus"],
    logger=wandb_logger,
    log_every_n_steps=trainer_params["log_every_n_steps"],
    val_check_interval=trainer_params["val_check_interval"],
    num_processes=trainer_params["num_processes"],
    max_epochs=trainer_params["max_epoch"],
    plugins=DDPPlugin(find_unused_parameters=False),
  )
  return trainer

# train
def train(trainer, model, train_loader, test_loader):
  """Function that start the training procedure of the model

  Args:
      trainer (pytorch_lightning.Trainer): Trainer that manages the training process of the model
      model (ErCaNet): Model that erases captions from images
      train_loader (DataLoader): Dataloader that serves the train dataset
      test_loader (DataLoader): Dataloader that serves the test dataset
  """
  trainer.fit(model, train_loader, test_loader)

# save model to file
def save_model_to_file(model, logger, model_save_path):
  """Function that saves the model to a file

  Args:
      model (ErCaNet): Model that erases captions from images
      logger (WandbLogger): Logger that logs information about the training progress
      model_save_path (str): A path defining where save current model.
  """
  print("##########################################")
  print(f"SAVING MODEL TO FILE: {logger.name}") 
  print("##########################################")
  torch.save(model.state_dict(), os.path.join(model_save_path, f'{logger.name}.pt'))
  print("##########################################")
  print(f"SAVED")
  print("##########################################")