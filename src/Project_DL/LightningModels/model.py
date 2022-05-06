import torch
import os
import torchvision.transforms.functional as fn
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F


class ErCaNet(pl.LightningModule):
	def __init__(self, my_name):
		"""Class that describes a model that erases captions from images.

		Args:
				my_name (string): model's name
		"""
		super().__init__()
		self.counter_of_val_images_saved = 0
		self.my_name = my_name
		self.cnn1 = nn.Sequential(
			nn.Conv2d(3,  16, (3, 3), 1, 1), nn.ReLU(),
			nn.Conv2d(16, 29, (3, 3), 1, 1), nn.ReLU()
		)
		self.cnn2 = nn.Sequential(
			nn.Conv2d(32, 32, (3, 3), 1, 1), nn.ReLU(),
			nn.Conv2d(32, 32, (3, 3), 1, 1), nn.ReLU()
		)
		self.cnn3 = nn.Sequential(
			nn.Conv2d(32, 16, (3, 3), 1, 1), nn.ReLU(),
			nn.Conv2d(16, 3,  (3, 3), 1, 1)
		)

	def forward(self, img):
		"""The forward method of the ErCaNet model.
		Erases captions from captioned image and fills in the blanks intelligently.

		Args:
				img (torch.Tensor): image with caption

		Returns:
				torch.Tensor: image with erased caption (hopefully)
		"""
		x = img
		c = self.cnn1(x)
		x = torch.cat([x, c], dim=1)
		x = x + self.cnn2(x)
		x = self.cnn3(x)
		return x

	def configure_optimizers(self):
		"""Method that configures the model's optimizer

		Returns:
				torch.optim.Optimizer: The model's optimizer
		"""
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		"""Method that describes one training step of the model.

		Args:
				train_batch (tuple(torch.Tensor, torch.Tensor)): Tuple of tensors describing captioned images and original images (uncaptioned)
				batch_idx (int): index of the batch

		Returns:
				torch.float32: MSE loss of the model on the training set
		"""
		dirty_img, orig_img = train_batch
		cleaned_img = self.forward(dirty_img)
		loss = F.mse_loss(orig_img, cleaned_img)
		self.log('train_loss', loss)
		return loss

	def validation_step(self, val_batch, batch_idx):
		"""Method that describes validation step of the model.
		Validates the performance of the model on unseen data. 

		Args:
				val_batch (tuple(torch.Tensor, torch.Tensor)): Tuple of tensors describing captioned images and original images (uncaptioned) the model has not seen in its training step
				batch_idx (int): index of the batch
		"""
		dirty_img, orig_img = val_batch
		cleaned_img = self.forward(dirty_img)
		loss = F.mse_loss(orig_img, cleaned_img)
		self.log('val_loss', loss)

		if not os.path.exists(os.path.join(f'data/training_overview/{self.my_name}')):
			os.mkdir(os.path.join(f'data/training_overview/{self.my_name}'))
			os.mkdir(os.path.join(f'data/training_overview/{self.my_name}/orig_imgs'))
			os.mkdir(os.path.join(f'data/training_overview/{self.my_name}/dirty_imgs'))
			os.mkdir(os.path.join(f'data/training_overview/{self.my_name}/cleaned_imgs'))

		if batch_idx % 300 == 0:
			fn.to_pil_image(orig_img[0, :, :, :]).save(
				os.path.join(f'data/training_overview/{self.my_name}/orig_imgs/orig_img_val_step_{self.counter_of_val_images_saved}.JPEG'), 
				'JPEG'
			)
			fn.to_pil_image(dirty_img[0, :, :, :]).save(
				os.path.join(f'data/training_overview/{self.my_name}/dirty_imgs/dirty_img_val_step_{self.counter_of_val_images_saved}.JPEG'), 
				'JPEG'
			)
			fn.to_pil_image(cleaned_img[0, :, :, :]).save(
				os.path.join(f'data/training_overview/{self.my_name}/cleaned_imgs/cleaned_img_val_step_{self.counter_of_val_images_saved}.JPEG'), 
				'JPEG'
			)
			self.counter_of_val_images_saved += 1
