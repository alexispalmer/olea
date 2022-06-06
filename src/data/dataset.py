import requests
import os
import csv
import pickle
import pandas as pd
import numpy as np

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
        self._data = None
        self.data_columns = None
        self.label_columns = None
        self.unique_labels = None

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


    def _download(self) : 
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

    def _find_unique_labels(self) :
        self.unique_labels = pd.unique(self._data[self.label_columns].values.ravel('K'))

    def _validate_predictions(self, dataset, submission, map=None) :

        if type(submission) not in [list, np.ndarray, pd.Series, tuple] : 
            raise TypeError('Expected submission to be in the form of one of the following types : '
            'list, np.ndarray, pd.Series, tuple')
        if map is None : 
            predictions = submission
        else : 
            predictions = self._map_submission_to_predictions(submission, map)

        if len(dataset) != len(predictions) : 
            raise ValueError('There is a mismatch in dataset of len() and prediction of len()'.format(len(dataset) , len(predictions)))

        if self.unique_labels is None : 
            self._find_unique_labels()

        for prediction in predictions : 
            if prediction not in self.unique_labels : 
                raise ValueError('You have submitted a prediction that does not have a basis in grountruth. Please provide a mapping.')

        return predictions


    def _map_submission_to_predictions(self, submission, map) :

        if self.unique_labels is None : 
            self._find_unique_labels() 

        if not all(sub in map.keys() for sub in submission) :
            raise ValueError('Problem encountered with one or more of your submissions.'
            'Please check if all submissions has a mapping.')

        if not all(label in self.unique_labels for label in map.values()) :
            raise ValueError('One or more of the mappings provided couldnt be traced back to a '
            'label in the groundtruth.')

        return [map[sub] for sub in submission]

    def data(self) : 
        return self._data[self.data_columns]

    def generator(self, batch_size) -> pd.DataFrame:
        
        start = 0
        end = batch_size

        while True : 

            if end < self._data.shape[0] :

                batch = self._data.loc[start:end, self.data_columns]
                start +=  batch_size
                end  = start + batch_size

            elif end >= self._data.shape[0] : 
                start, end = self._data.shape[0] - batch_size, self._data.shape[0]
                batch = self._data.loc[start:end, self.data_columns]
                start, end = 0, batch_size
                
            yield batch

    def info(self) -> None : 
        pass

    def submit(self, dataset:pd.DataFrame, submission:iter, map:dict=None) -> None : 

        valid_predictions = self._validate_predictions(dataset, submission, map)
        dataset['preds'] = valid_predictions
        print(dataset)
        # To Marie's Analysis engine
        pass

