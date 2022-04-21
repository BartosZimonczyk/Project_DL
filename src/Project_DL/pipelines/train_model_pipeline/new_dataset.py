import pickle
import os
import torch
import random
import numpy as np
import torchvision.transforms.functional as fn

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image, ImageFont, ImageDraw

from matplotlib.pyplot import imshow, show


class BatchesImagesDataset(Dataset):
    def __init__(self, data_path='data/all_clean_batches', font_path='data/fonts', resize_up_to=None, true_randomness=False):
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
        self.IMG_SIZE = 64
        self.batch_sizes = self._get_batch_sizes()
        self.current_item = 0
        self.current_batch_name = 'train_data_batch_1'
        self.current_batch_data = self.unpickle(os.path.join(self.data_path, self.current_batch_name))['data']
        self._tune_current_data()

    def _tune_current_data(self):
        img_size2 = self.IMG_SIZE * self.IMG_SIZE
        self.current_batch_data = np.dstack((
            self.current_batch_data[:, :img_size2],
            self.current_batch_data[:, img_size2:2*img_size2],
            self.current_batch_data[:, 2*img_size2:]
        ))
        self.current_batch_data = self.current_batch_data.reshape((
            self.current_batch_data.shape[0],
            self.IMG_SIZE,
            self.IMG_SIZE,
            3
        ))
    
    def _get_batch_size(self, batch_name):
        if batch_name in os.listdir(self.data_path):
            d = self.unpickle(os.path.join(self.data_path, batch_name))
            _, y, _ = d['data'], d['labels'], d['mean']
            return len(y)
        else:
            raise ValueError(f"There is no '{batch_name}' file in the '{self.data_path}' folder")
    
    def _get_batch_sizes(self):
        batches = os.listdir(self.data_path)
        batch_sizes = {batch: self._get_batch_size(batch) for batch in batches}
        
        return batch_sizes

    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as f:
            dict = pickle.load(f)

        return dict
    
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
        batches = sorted(os.listdir(self.data_path))
        prev_size = 0

        if index >= len(self):
            raise ValueError(f"{index} is out of index.")

        for batch in batches:
            if index < prev_size + self.batch_sizes[batch]:
                if batch == self.current_batch_name:
                    pass
                else:
                    self.current_batch_name = batch
                    self.current_batch_data = self.unpickle(os.path.join(self.data_path, batch))['data']
                    self._tune_current_data()
                data = self.current_batch_data
            else:
                prev_size += self.batch_sizes[batch]
        
        image = Image.fromarray(
            data[index-prev_size, :, :, :]
        )

        if self.resize_up_to is not None:
            image = fn.resize(image, size=(self.resize_up_to, self.resize_up_to))

        image_with_caption = self.add_random_text(image, index)

        return (ToTensor()(image), ToTensor()(image_with_caption))

if __name__ == "__main__":
    ds = BatchesImagesDataset(resize_up_to=256)
    print('done')
    image_0 = ds[0]
    imshow(image_0[0].permute(1, 2, 0))
    show()
    imshow(image_0[1].permute(1, 2, 0))
    show()