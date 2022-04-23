import os
import torch
import random
import numpy as np
import torchvision.transforms.functional as fn

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image, ImageFont, ImageDraw

from matplotlib.pyplot import imshow, show


class UnpickledImagesDataset(Dataset):
    def __init__(self, max_batches=5, data_path='data/all_unpickle', font_path='data/fonts', resize_up_to=None, true_randomness=False):
        """
        A dataset class build on top of PyTorch builtin Dataset class.
        One should use this class while building dataloaders, if raw data is in data_path
        in form of .JPEG images. Files in the mentioned folder should be named 
        "image_{integers, starting from 0}.JEPG". 
        """
        self.max_batches = max_batches
        self.data_path = data_path
        self.font_path = font_path
        self.resize_up_to = resize_up_to
        self.true_randomness = true_randomness
        self.LOREM_IPSUM = \
            """
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur vel dui placerat, elementum augue at, maximus ipsum. Vivamus eros enim, aliquet ullamcorper mauris at, faucibus malesuada nunc. Donec vel purus iaculis, dapibus felis vel, malesuada dui. Sed condimentum est id dapibus convallis. Maecenas vestibulum malesuada enim, id faucibus augue dictum at. Ut at condimentum mi. Cras placerat risus vel lorem pretium, non efficitur eros condimentum. Donec pretium ut arcu at maximus. Phasellus vehicula mattis sapien, ut placerat quam faucibus ac. Phasellus porta urna ex, quis ultricies mi auctor sed. Duis mattis faucibus est ac hendrerit.
            Ut tempor nunc ac purus dapibus, et placerat urna rhoncus. Suspendisse blandit vitae leo et laoreet. Cras efficitur nunc neque, et malesuada mi maximus in. Mauris eleifend tortor id felis viverra, et aliquet diam lobortis. Duis gravida urna ut tortor interdum, ut ultrices nisl iaculis. Etiam mi ex, commodo a finibus at, dapibus at odio. Ut dignissim est velit. Mauris et suscipit risus. Etiam ornare odio et velit bibendum lacinia.
            In hendrerit tempus nisl, a sollicitudin massa maximus vel. Phasellus luctus non nulla sit amet feugiat. Aliquam sit amet eros orci. Vestibulum non mattis massa. Maecenas vehicula luctus leo sed aliquam. Nam non est sit amet ante dictum bibendum. Nam quis accumsan odio. Curabitur posuere ex tellus, euismod pellentesque mauris tincidunt vel. Phasellus auctor ante nunc, et tempus eros placerat in. Duis tristique tempus purus ut placerat.
            Suspendisse sagittis egestas rutrum. Fusce vestibulum dolor non sem maximus sollicitudin. Aenean ac metus neque. Donec sed massa finibus, venenatis tortor non, tempor ex. Vestibulum imperdiet suscipit nibh, eget feugiat tortor interdum eu. Fusce non posuere ex. Praesent vel felis quis lacus varius ultricies. Donec id quam ut orci convallis sollicitudin. Curabitur magna leo, pharetra quis sapien in, eleifend lobortis eros. Suspendisse ut purus sem. Nam a vehicula sapien. Maecenas id pulvinar velit. Aenean lobortis felis id risus porttitor, in sagittis enim facilisis.
            Vestibulum efficitur cursus metus, vel faucibus sem venenatis ut. Aenean luctus felis turpis, sed molestie leo varius eu. Cras posuere sollicitudin gravida. Duis ut blandit justo. Donec leo diam, euismod a aliquet sit amet, dictum eget turpis. Suspendisse lacus diam, porta sagittis aliquam ut, laoreet ullamcorper nisl. Pellentesque semper pellentesque ipsum sed consectetur. In posuere nisi at est accumsan, sit amet viverra felis egestas. Donec placerat luctus faucibus.
            """
        self.batch_names = sorted(os.listdir(self.data_path))[:self.max_batches]
        self.batch_sizes = self._get_batch_sizes()
        self.cumulitve_batch_sizes = np.cumsum([v for v in self.batch_sizes.values()])

    def _get_batch_size(self, batch_name):
        if batch_name in self.batch_names:
            return len(os.listdir(os.path.join(self.data_path, batch_name)))
        else:
            raise ValueError(f"There is no '{batch_name}' file in the '{self.data_path}' folder")
    
    def _get_batch_sizes(self):
        batch_sizes = {batch: self._get_batch_size(batch) for batch in self.batch_names}
        
        return batch_sizes
    
    def get_random_font(self, seed=0):
        if not self.true_randomness:
            random.seed(seed)
        
        if os.path.isdir(self.font_path):
            possible_fonts = [f for f in os.listdir(self.font_path) if str(f).split('.')[1] == 'ttf']
            font = random.choice(possible_fonts)
        return font

    def add_random_text(self, image, seed=0):
        if not self.true_randomness:
            random.seed(seed)
        
        my_image = image.convert('RGB')
        w = my_image.width
        h = my_image.height
        font_size = random.randint(int(h/15.0), int(h/7.0))
        title_font = ImageFont.truetype(self.get_random_font(seed), font_size)
        sentence = self.LOREM_IPSUM.split(".")[random.randint(0,53)]
        sentence = sentence.split()
        t_len = random.randint(min(2, len(sentence)),min(5, len(sentence)))
        s_beg = random.randint(0, len(sentence)-t_len)
        text = (" ").join(sentence[s_beg:s_beg+t_len])
        image_editable = ImageDraw.Draw(my_image)
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        left = (random.randint(int(-0.1*w), int(0.25*w)))
        top = (random.randint(int(0.05*h), int(0.85*h)))
        image_editable.text((left, top), text, color, font=title_font)
        
        return my_image

    def __len__(self):
        return sum(self.batch_sizes.values())

    def __getitem__(self, index):
        if index >= len(self):
            raise ValueError(f"Index {index} is out of bounds.")
        
        use_batch_name, batch_no = None, 0
        for k, batch_name in enumerate(self.batch_names):
            if index < self.cumulitve_batch_sizes[k]:
                use_batch_name, batch_no = batch_name, k
                break
            else:
                pass
        
        if k == 0:
            image_path = os.path.join(self.data_path, use_batch_name, f"image_{index}.JPEG")
        else:
            image_path = os.path.join(self.data_path, use_batch_name, f"image_{index-self.cumulitve_batch_sizes[k-1]}.JPEG")

        if os.path.exists(image_path):
            image = Image.open(image_path)
        else:
            raise ValueError(f"There is not image in path: {image_path}")
        
        if self.resize_up_to is not None:
            image = fn.resize(image, size=(self.resize_up_to, self.resize_up_to))
        
        # we can omit seed input in below method, to have truly random output
        image_with_caption = self.add_random_text(image, index)

        return (ToTensor()(image), ToTensor()(image_with_caption))

if __name__ == "__main__":
    ds = UnpickledImagesDataset(resize_up_to=256)
    print(len(ds))
    print(ds.get_random_font())
    imshow(ds[130000][0].permute(1, 2, 0))
    show()
    imshow(ds[130000][1].permute(1, 2, 0))
    show()