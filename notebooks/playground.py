import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch

from matplotlib.pyplot import imshow, show
from torchvision.transforms import ToTensor
from PIL import Image
from Project_DL.LightningModels.model import ErCaNet


model_2 = ErCaNet("Playground_2")
model_2.load_state_dict(torch.load('models/CaptionEraseBZ-GPU-TheThird.pt'))
model_2.eval()

img_1 = ToTensor()(Image.open('notebooks/orig_img_val_step_0.JPEG'))

cleaned_img_1 = model_2.forward(img_1[None, :])

imshow(img_1.permute(1, 2, 0))
show()

imshow(cleaned_img_1.detach().numpy()[0].transpose(1, 2, 0))
show()