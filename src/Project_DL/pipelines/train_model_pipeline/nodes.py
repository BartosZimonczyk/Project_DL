"""
This is a boilerplate pipeline 'train_model_pipeline'
generated using Kedro 0.17.7
"""

from torch.utils.data import random_split

from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import node, Pipeline
from kedro.runner import SequentialRunner
from Project_DL.pipelines.train_model_pipeline.model import ErCaNet
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from dataloaders import DataModuleClass


data_catalog = DataCatalog({"dataset": MemoryDataSet()})
data_path = 'data/all_clean'
font_path = 'data/fonts'
true_randomness = False
batch_size = 1
shuffle_in_loader = True

# data
def load_dataset():
	dataset = DataModuleClass(data_path, font_path, true_randomness)
	dataset.setup()
	train_loader = dataset.train_dataloader(batch_size, shuffle_in_loader) # loads clean images from disk, adds captions on-the-fly
	test_loader = dataset.test_dataloader(batch_size, shuffle_in_loader)  # loads clean images from disk, adds captions on-the-fly
	val_loader = dataset.val_dataloader(batch_size, shuffle_in_loader)   # loads clean images from disk, adds captions on-the-fly
	return train_loader, test_loader, val_loader

load_dataset_node = node(load_dataset, inputs=None, outputs=["train_loader", "test_loader", "val_loader"])

# model
def get_model():
	model = ErCaNet()
	return model
 
get_model_node = node(get_model, inputs=None, outputs="model")

#logger
def get_logger():
	wandb_logger = WandbLogger(project="test-project")
	return wandb_logger

get_logger_node = node(get_logger, inputs=None, outputs="logger")


# trainer
def get_trainer(wandb_logger):
	trainer = Trainer(logger=wandb_logger)
	return trainer

get_trainer_node = node(get_trainer, inputs="logger", outputs="trainer")

# train
def train(trainer, model, train_loader, test_loader):
	trainer.fit(model, train_loader, test_loader)
 
 
train_node = node(train, inputs=["trainer", "model", "train_loader", "test_loader"], outputs="")
