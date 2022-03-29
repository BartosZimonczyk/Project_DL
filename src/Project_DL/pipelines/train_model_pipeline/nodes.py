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



data_catalog = DataCatalog({"dataset": MemoryDataSet()})

train_clean_path = "../../../../data/train_clean"
test_clean_path = "../../../../data/test_clean"
test_with_caption_path = "../../../../data/test_with_caption"
val_clean_path = "../../../../data/val_clean"
val_with_caption_path = "../../../../data/val_with_caption"

# data
def load_dataset():
		# dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
		# mnist_train, mnist_val = random_split(dataset, [55000, 5000])

		# train_loader = DataLoader(mnist_train, batch_size=32)
		# val_loader = DataLoader(mnist_val, batch_size=32)
		# return train_loader, val_loader
  train_loader = None # loads clean images from disk, adds captions on-the-fly
  test_loader = None  # loads both clean & captioned images from disk
  val_loader = None   # loads both clean & captioned images from disk
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

