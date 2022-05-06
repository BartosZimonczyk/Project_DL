import pytorch_lightning as pl
import numpy as np

from torch.utils.data import random_split, DataLoader
from Project_DL.DataClasses.newest_dataset import UnpickledImagesDataset

class DataModuleClass(pl.LightningDataModule):
    def __init__(self, max_batches=5, data_path='data/all_clean', font_path='data/fonts', resize_up_to=None, true_randomness=False, transform=None):
        """
        A dataloader class that build on top of the Pytorch Lighting data module class. It utilizes the UnpickledImagesDataset class.

        Args:
            max_batches (int, optional): Number of pickled batches to use (It depends on the data that are downloaded). Defaults to 5.
            data_path (str, optional): Path to the images data. Defaults to 'data/all_clean'.
            font_path (str, optional): Path to the font data. Defaults to 'data/fonts'.
            resize_up_to (int, optional): Number of pixels that the images should be resized to, if None no transformation is done. Defaults to None.
            true_randomness (bool, optional): Boolean value to decide wheather the text on the images should be truly random or reproducible random. Defaults to False.
            transform (_type_, optional): Depricated. Transforms to apply on the images. Defaults to None.
        """
        super().__init__(self)
        self.max_batches = max_batches
        self.data_path = data_path
        self.font_path = font_path
        self.resize_up_to = resize_up_to
        self.true_randomness = true_randomness
        self.transform = transform
    
    def prepare_data(self):
        # Define steps that should be done
        # on only one GPU, like getting data.
        """
        A method to prepare dataset. Shouldnt be run by hand, by user.
        """
        self.dataset = UnpickledImagesDataset(
            self.max_batches,
            self.data_path,
            self.font_path,
            self.resize_up_to,
            self.true_randomness,
        )
    
    def setup(self, proportions=(0.9, 0.05, 0.05), stage=None):
        # Define steps that should be done on 
        # every GPU, like splitting data, applying
        # transform etc.
        """
        A method that setup train/val/test split with proportions set in propotions atribute.

        Args:
            proportions (tuple, optional): How to split the data into train/val/test sets. Defaults to (0.9, 0.05, 0.05).
            stage (_type_, optional): Defaults to None.
        """
        self.prepare_data()
        n = len(self.dataset)
        p_train, p_test, p_val = proportions
        train_n = np.floor(n*p_train).astype(np.int_)
        test_n = np.floor(n*p_test).astype(np.int_)
        lenghts = [train_n, test_n, n-train_n-test_n]
        self.train_set, self.test_set, self.val_set = random_split(self.dataset, lenghts)
    
    def train_dataloader(self, batch_size=1, shuffle=False, num_workers=1):
        # Return DataLoader for Training Data here
        """
        A method that return the dataloader of train dataset.

        Args:
            batch_size (int, optional): Number of images in one batch. Defaults to 1.
            shuffle (bool, optional): Boolean value wheater, batch should be shuffled. Defaults to False.
            num_workers (int, optional): Number of workers passed to the torch.utils.data.DataLoader class. Defaults to 1.

        Returns:
            DataLoader: A train dataset DataLoader. 
        """
        return DataLoader(
            dataset=self.train_set,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=True,
            drop_last=True,
        )
    
    def val_dataloader(self, batch_size=1, shuffle=False, num_workers=1):
        # Return DataLoader for Validation Data here
        """
        A method that return the dataloader of validation dataset.

        Args:
            batch_size (int, optional): Number of images in one batch. Defaults to 1.
            shuffle (bool, optional): Boolean value wheater, batch should be shuffled. Defaults to False.
            num_workers (int, optional): Number of workers passed to the torch.utils.data.DataLoader class. Defaults to 1.

        Returns:
            DataLoader: A validation dataset DataLoader. 
        """
        return DataLoader(
            dataset=self.val_set,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=True,
            drop_last=True,
        )
    
    def test_dataloader(self, batch_size=1, shuffle=False, num_workers=1):
        # Return DataLoader for Testing Data here
        """
        A method that return the dataloader of test dataset.

        Args:
            batch_size (int, optional): Number of images in one batch. Defaults to 1.
            shuffle (bool, optional): Boolean value wheater, batch should be shuffled. Defaults to False.
            num_workers (int, optional): Number of workers passed to the torch.utils.data.DataLoader class. Defaults to 1.

        Returns:
            DataLoader: A test dataset DataLoader. 
        """
        return DataLoader(
            dataset=self.test_set,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=True,
            drop_last=True,
        )