from typing import List, Union
import pandas as pd
import numpy as np

from olea.data.dso import DatasetSubmissionObject

class Dataset(object) : 

    def __init__(self, data:pd.DataFrame=None, 
                features=Union[str, List[str]], 
                gold_column=str,
                text_column=str) -> None:
        """Initializes the dataset instance. 

        The requested dataset is downloaded and saved on the local system. You can choose where the dataset 
        should be saved by providing a dataset_save_dir.

        Args:
            dataset_save_dir (str, optional): _description_. Defaults to '~/cold/'.
        """
        self.URL = None
        self.dataset_name = None
        self.shape = None
        self.description = None
        
        self.features = None
        self.gold_label = None
        self.text_column = None
        
        self.unique_labels = None
        self._data = None
        
        self._load_data(data, features, gold_column, text_column)

    def __call__(self) -> None:
        pass

    def __iter__(self) -> None: 
        pass

    def __str__(self) -> str:
        pass

    def _load_data(self, data:pd.DataFrame, 
                features:Union[str, List[str]], 
                gold_column:str,
                text_column:str) -> None:
        self.features = features
        self.gold_column = gold_column
        self.text_column = text_column
        self._data = data

    def _find_unique_labels(self) :
        self.unique_labels = pd.unique(self._data[self.gold_column].values.ravel('K'))

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
        return self._data[self.features]

    def generator(self, batch_size) -> pd.DataFrame:
        
        batch_size -= 1 # Pandas indexes weird
        start = 0
        end = batch_size

        while True : 

            if end < self._data.shape[0] :

                batch = self._data.loc[start:end, self.features]
                start +=  batch_size + 1 # Once again, Pandas indexes weirdly
                end  = start + batch_size

            elif end >= self._data.shape[0] : 
                start, end = self._data.shape[0] - batch_size, self._data.shape[0]
                batch = self._data.loc[start:end, self.features]
                start, end = 0, batch_size
                
            yield batch

    def info(self) -> None : 
        pass

    def submit(self, batch:pd.DataFrame, predictions:iter, map:dict=None) -> None : 

        valid_predictions = self._validate_predictions(batch, predictions, map)
        submission_df = self._data.loc[self._data.index.isin(batch.index.values)]
        submission_df['preds'] = valid_predictions

        submission_object = DatasetSubmissionObject(submission_df, self)

        return submission_object

