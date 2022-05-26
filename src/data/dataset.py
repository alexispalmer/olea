import requests
import os
import csv
import pickle

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
        self.shape = None
        self.description = None
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

        dataset_basedir = os.path.basename(self.dataset_save_dir)

        if not os.path.isdir(dataset_basedir) :
            raise ValueError('Unable to find the base directory {} for dataset directory {}'.format(dataset_basedir, 
                                                                                                self.dataset_save_dir))

        if not os.path.isdir(self.dataset_save_dir) : 
            os.mkdir(self.dataset_save_dir)
            return False

        return True


    def _download(self) -> bool : 
        """Base class that downloads the dataset from an online path. 

        Args:
            dataset_name (str): Name of the dataset from the repo

        Returns:
            bool: _description_
        """
        r = requests.get(self.URL)
        lines = r.content.split('\n')
        data = csv.reader(lines, delimiter='\t')

        with open(os.path.join(self.dataset_save_dir , self.BASEURL) , 'wb') as f : 
            f.write(data)


    def data(self) : 
        pass

    def generator(self) -> None : 
        pass

    def info(self) -> None : 
        pass

    def analyze(self) -> None : 
        pass

