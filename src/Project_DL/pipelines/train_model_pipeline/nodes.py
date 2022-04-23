"""
This is a boilerplate pipeline 'train_model_pipeline'
generated using Kedro 0.17.7
"""

import wandb
import torch
import os

from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import node, Pipeline
from kedro.runner import SequentialRunner
from Project_DL.pipelines.train_model_pipeline.model import ErCaNet
from pytorch_lightning import Trainer

from Project_DL.DataClasses.dataloaders import DataModuleClass


wandb.init(project="ErCaNet", entity="coldteam")

data = {}
memory_dataset = MemoryDataSet(data)
data_catalog = DataCatalog({"dataset": memory_dataset})
max_batches = 5
data_path = 'data/all_unpickle'
font_path = 'data/fonts'
model_save_path = 'models/'
true_randomness = False
resize_up_to = 256
batch_size = 64
shuffle_in_loader = True

# data
def load_dataset():
  dataset = DataModuleClass(max_batches, data_path, font_path, resize_up_to, true_randomness)
  dataset.setup()
  train_loader = dataset.train_dataloader(batch_size, shuffle_in_loader) # loads clean images from disk, adds captions on-the-fly
  test_loader = dataset.test_dataloader(batch_size, shuffle_in_loader)  # loads clean images from disk, adds captions on-the-fly
  val_loader = dataset.val_dataloader(batch_size, shuffle_in_loader)   # loads clean images from disk, adds captions on-the-fly
  return train_loader, test_loader, val_loader

# model
def get_model():
  model = ErCaNet()
  wandb.watch(model)
  return model

# trainer
def get_trainer():
  print("##########################################")
  print("TRAINER LOADING")
  print("##########################################")
  trainer = Trainer(accelerator='auto')
  print("##########################################")
  print("TRAINER LOADED")
  print("##########################################")
  return trainer

# train
def train(trainer, model, train_loader, test_loader):
  print("##########################################")
  print("TRAINING")
  print("##########################################")
  trainer.fit(model, train_loader, test_loader)
  print("##########################################")
  print("TRAINED")
  print("##########################################")

# save model to file
def save_model_to_file(model):
  print("##########################################")
  print(f"SAVING MODEL TO FILE: {wandb.run.name}")
  print("##########################################")
  torch.save(model.state_dict(), os.path.join(model_save_path, f'{wandb.run.name}.pt'))
  print("##########################################")
  print(f"SAVED")
  print("##########################################")