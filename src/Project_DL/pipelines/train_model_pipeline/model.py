import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl

class ErCaNet(pl.LightningModule):
	def __init__(self):
		super().__init__()
		self.cnn = nn.Sequential(
    	nn.Conv2d(3, 16, (3, 3), 1, 1), nn.ReLU(),
			nn.Conv2d(16, 32, (3, 3), 1, 1), nn.ReLU(),
			# nn.Conv2d(32, 64, (3, 3), 1, 1), nn.ReLU(),
			# nn.Conv2d(64, 32, (3, 3), 1, 1), nn.ReLU(),
    	nn.Conv2d(32, 16, (3, 3), 1, 1), nn.ReLU(),
			nn.Conv2d(16, 3, (3, 3), 1, 1)
		)

	def forward(self, img):
		return img + self.cnn(img)

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		dirty_img, orig_img = train_batch
		cleaned_img = self.forward(dirty_img)
		loss = F.mse_loss(orig_img, cleaned_img)
		self.log('train_loss', loss)
		return loss

	def validation_step(self, val_batch, batch_idx):
		dirty_img, orig_img = val_batch
		cleaned_img = self.forward(dirty_img)
		loss = F.mse_loss(orig_img, cleaned_img)
		self.log('val_loss', loss)