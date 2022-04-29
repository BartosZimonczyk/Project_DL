"""
This is a boilerplate pipeline 'train_model_pipeline'
generated using Kedro 0.17.7
"""

import torch
import os

from kedro.io import DataCatalog, MemoryDataSet
from pytorch_lightning.loggers import WandbLogger
from Project_DL.pipelines.train_model_pipeline.model import ErCaNet
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin

from Project_DL.DataClasses.dataloaders import DataModuleClass


data = {}
memory_dataset = MemoryDataSet(data)
data_catalog = DataCatalog({"dataset": memory_dataset})
max_batches = 3
data_path = 'data/all_unpickle'
font_path = 'data/fonts'
model_save_path = 'models/'
true_randomness = False
resize_up_to = 256
batch_size = 32
loader_workers = 8

# data
def load_dataset():
  dataset = DataModuleClass(max_batches, data_path, font_path, resize_up_to, true_randomness)
  dataset.setup()
  train_loader = dataset.train_dataloader(batch_size, shuffle = True, num_workers=loader_workers) # loads clean images from disk, adds captions on-the-fly
  test_loader = dataset.test_dataloader(batch_size, shuffle = False, num_workers=loader_workers)  # loads clean images from disk, adds captions on-the-fly
  val_loader = dataset.val_dataloader(batch_size, shuffle = False, num_workers=loader_workers)    # loads clean images from disk, adds captions on-the-fly
  return train_loader, test_loader, val_loader

# model
def get_model(logger):
  model = ErCaNet(logger.name)
  # wandb.watch(model)
  return model

def get_logger():
  wandb_logger = WandbLogger(name='CaptionEraseBZ-GPU-TheSecond', project='ErCaNet')
  return wandb_logger

# trainer
def get_trainer(wandb_logger):
  trainer = Trainer(
    accelerator='gpu',
    gpus=1,
    logger=wandb_logger,
    log_every_n_steps=10,
    val_check_interval=0.1,
    num_processes=1,
    max_epochs=5,
    plugins=DDPPlugin(find_unused_parameters=False),
  )
  return trainer

# train
def train(trainer, model, train_loader, test_loader):
  trainer.fit(model, train_loader, test_loader)

# save model to file
def save_model_to_file(model, logger):
  print("##########################################")
  print(f"SAVING MODEL TO FILE: {logger.name}") 
  print("##########################################")
  torch.save(model.state_dict(), os.path.join(model_save_path, f'{logger.name}.pt'))
  print("##########################################")
  print(f"SAVED")
  print("##########################################")