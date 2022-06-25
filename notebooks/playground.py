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


model_2 = torch.load('models/CaptionEraseBZ-GPU-TheSixth.pt')
model_2.eval()

img_1 = ToTensor()(Image.open('notebooks/dirty_img_val_step_1.JPEG'))

cleaned_img = model_2.forward(img_1[None, :])

imshow(img_1.permute(1, 2, 0))
print(img_1.permute(1, 2, 0))
show()

min_val = torch.min(torch.flatten(cleaned_img))
max_val = torch.max(torch.flatten(cleaned_img))
output_img = (cleaned_img- min_val) / max_val

print(output_img)
pil_img = fn.to_pil_image(output_img[0, :, :, :])
pil_img.show()

# np_img = cleaned_img.detach().numpy()[0, :, :, :].transpose(1, 2, 0)
# imshow(np_img)
# print(np_img)
# show()

# min_val = np.amin(np_img.ravel())
# max_val = np.amax(np_img.ravel()) - min_valtorch.save(model, os.path.join(model_save_path, f'{logger.name}.pt'))
# output_img = (np_img - min_val)/max_val 

# imshow(output_img)
# print(output_img)
# show()