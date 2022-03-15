"""
This is a boilerplate pipeline 'train_model_pipeline'
generated using Kedro 0.17.7
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import node, Pipeline
from kedro.runner import SequentialRunner


class LitAutoEncoder(pl.LightningModule):
	def __init__(self):
		super().__init__()
		self.encoder = nn.Sequential(
      nn.Linear(28 * 28, 64),
      nn.ReLU(),
      nn.Linear(64, 3))
		self.decoder = nn.Sequential(
      nn.Linear(3, 64),
      nn.ReLU(),
      nn.Linear(64, 28 * 28))

	def forward(self, x):
		embedding = self.encoder(x)
		return embedding

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		x, y = train_batch
		x = x.view(x.size(0), -1)
		z = self.encoder(x)    
		x_hat = self.decoder(z)
		loss = F.mse_loss(x_hat, x)
		self.log('train_loss', loss)
		return loss

	def validation_step(self, val_batch, batch_idx):
		x, y = val_batch
		x = x.view(x.size(0), -1)
		z = self.encoder(x)
		x_hat = self.decoder(z)
		loss = F.mse_loss(x_hat, x)
		self.log('val_loss', loss)


data_catalog = DataCatalog({"dataset": MemoryDataSet()})

# data
def load_dataset():
		dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
		mnist_train, mnist_val = random_split(dataset, [55000, 5000])

		train_loader = DataLoader(mnist_train, batch_size=32)
		val_loader = DataLoader(mnist_val, batch_size=32)
		return train_loader, val_loader

load_dataset_node = node(load_dataset, inputs=None, outputs=["train_loader", "val_loader"])

# model
def get_model():
		model = LitAutoEncoder()
		return model
 
get_model_node = node(get_model, inputs=None, outputs="model")

#logger
def get_logger():
		wandb_logger = WandbLogger(project="test-project")
		return wandb_logger

get_logger_node = node(get_logger, inputs=None, outputs="logger")


# trainer
def get_trainer(wandb_logger):
		trainer = pl.Trainer(logger=wandb_logger)
		return trainer

get_trainer_node = node(get_trainer, inputs="logger", outputs="trainer")

# train
def train(trainer, model, train_loader, val_loader):
		trainer.fit(model, train_loader, val_loader)
 
 
train_node = node(train, inputs=["trainer", "model", "train_loader", "val_loader"], outputs="")

