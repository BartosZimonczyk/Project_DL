import pytorch_lightning as pl
import numpy as np

from torch.utils.data import random_split, DataLoader
from Project_DL.DataClasses.new_dataset import BatchesImagesDataset

class DataModuleClass(pl.LightningDataModule):
    def __init__(self, data_path='data/all_clean', font_path='data/fonts', resize_up_to=None, true_randomness=False, transform=None):
        #Define required parameters here
        super().__init__(self)
        self.data_path = data_path
        self.font_path = font_path
        self.resize_up_to = resize_up_to
        self.true_randomness = true_randomness
        self.transform = transform
    
    def prepare_data(self):
        # Define steps that should be done
        # on only one GPU, like getting data.
        self.dataset = BatchesImagesDataset(
            self.data_path,
            self.font_path,
            self.resize_up_to,
            self.true_randomness,
        )
    
    def setup(self, proportions=(0.7, 0.15, 0.15), stage=None):
        # Define steps that should be done on 
        # every GPU, like splitting data, applying
        # transform etc.
        self.prepare_data()
        n = len(self.dataset)
        p_train, p_test, p_val = proportions
        train_n = np.floor(n*p_train).astype(np.int_)
        test_n = np.floor(n*p_test).astype(np.int_)
        lenghts = [train_n, test_n, n-train_n-test_n]
        self.train_set, self.test_set, self.val_set = random_split(self.dataset, lenghts)
    
    def train_dataloader(self, batch_size=1, shuffle=False):
        # Return DataLoader for Training Data here
        return DataLoader(
            dataset=self.train_set,
            batch_size=batch_size,
            shuffle=shuffle
        )
    
    def val_dataloader(self, batch_size=1, shuffle=False):
        # Return DataLoader for Validation Data here
        return DataLoader(
            dataset=self.val_set,
            batch_size=batch_size,
            shuffle=shuffle
        )
    
    def test_dataloader(self, batch_size=1, shuffle=False):
        # Return DataLoader for Testing Data here
        return DataLoader(
            dataset=self.test_set,
            batch_size=batch_size,
            shuffle=shuffle
        )