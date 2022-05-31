import requests
import os
import csv
import pickle
import pandas as pd

class Dataset(object) : 

    def __init__(self, 
                dataset_save_dir:str='~/cold/') -> None:
        """Initializes the dataset instance. 

        The requested dataset is downloaded and saved on the local system. You can choose where the dataset 
        should be saved by providing a dataset_save_dir.

        Args:
            dataset_save_dir (str, optional): _description_. Defaults to '~/cold/'.
        """
        
        self.dataset_save_dir = dataset_save_dir

        self.URL = None
        self.BASEURL = None
        self.dataset_name = None
        self.dataset_path = None
        self.shape = None
        self.description = None
        self.i = None
        self._data = None
        pass

    def __call__(self) -> None:
        pass

    def __iter__(self) -> None: 
        pass

    def __str__(self) -> str:
        pass

    
    def _dataset_file_exists(self) -> bool : 

        if self.BASEURL in os.listdir(self.dataset_save_dir) : 
            return True

        return False


    def _save_dir_exists(self) -> bool : 

        if not os.path.isdir(self.dataset_save_dir) : 
            os.mkdir(self.dataset_save_dir)

        return True


    def _download(self) -> csv.DictReader : 
        """Base class that downloads the dataset from an online path. 

        Args:
            dataset_name (str): Name of the dataset from the repo

        Returns:
            bool: _description_
        """
        pass

    def _load_data(self) -> None:

        if self._save_dir_exists() : 
            if self._dataset_file_exists() : 
                with open(self.dataset_path, 'rb') as f : 
                    self._data = pickle.load(f)

            else : 
                self._data = self._download() 


    def data(self) : 
        pass

    def generator(self) -> None : 
        pass

    def info(self) -> None : 
        pass

    def submit(self) -> None : 
        pass

