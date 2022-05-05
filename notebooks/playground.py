import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch

from matplotlib.pyplot import imshow, show
from torchvision.transforms import ToTensor
from PIL import Image
from src.Project_DL.pipelines.train_model_pipeline.model import ErCaNet


model_2 = ErCaNet("Playground_2")
model_2.load_state_dict(torch.load('models/CaptionEraseBZ-GPU-TheThird.pt'))
model_2.eval()

img_1 = ToTensor()(Image.open('notebooks/meme1.jpg'))

cleaned_img_1 = model_2.forward(img_1)

imshow(img_1.permute(1, 2, 0))
show()

imshow(cleaned_img_1.permute(1, 2, 0))
show()