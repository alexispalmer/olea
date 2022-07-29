import pandas as pd

from typing import List, Tuple


class DatasetSubmissionObject(object) : 

    def __init__(self, 
                submission_df:pd.DataFrame, 
                dataset_object:object) : 
        
        for key, value in dataset_object.__dict__.items() : 
            setattr(self, key, value)

        self.submission = submission_df
        self.prediction_column = 'preds'

    def data(self) : 
        return self.submission

    def filter_submission(self, on:str, filter:callable, **kwargs)  : 
        pass
