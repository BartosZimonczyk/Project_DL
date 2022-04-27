import torch
import os
import torchvision.transforms.functional as fn
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F


class ErCaNet(pl.LightningModule):
	def __init__(self, my_name):
		super().__init__()
		self.my_name = my_name
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
		if batch_idx % 100 == 0:
			if not os.path.exists(os.path.join(f'data/training_overview/{self.my_name}')):
				os.mkdir(os.path.join(f'data/training_overview/{self.my_name}'))
			fn.to_pil_image(orig_img[0, :, :, :]).save(os.path.join(f'data/training_overview/{self.my_name}/orig_img_val_step_{batch_idx}.JPEG'), 'JPEG')
			fn.to_pil_image(dirty_img[0, :, :, :]).save(os.path.join(f'data/training_overview/{self.my_name}/dirty_img_val_step_{batch_idx}.JPEG'), 'JPEG')
			fn.to_pil_image(cleaned_img[0, :, :, :]).save(os.path.join(f'data/training_overview/{self.my_name}/cleaned_img_val_step_{batch_idx}.JPEG'), 'JPEG')