import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
import numpy as np
import torchvision.transforms.functional as fn

from matplotlib.pyplot import imshow, show
from torchvision.transforms import ToTensor
from PIL import Image
from Project_DL.LightningModels.model import ErCaNet



model = ErCaNet.load_from_checkpoint("CaptionEraseBZ-GPU-TheThird/version_None/checkpoints/epoch=4-step=36023.ckpt", my_name="Please work")
model.eval()

img_1 = ToTensor()(Image.open('notebooks/dirty_img_val_step_0.JPEG'))

cleaned_img = model.forward(img_1[None, :])

imshow(img_1.permute(1, 2, 0))
show()

np_img = cleaned_img.detach().numpy()[0, :, :, :].transpose(1, 2, 0)
imshow(np_img)
show()
