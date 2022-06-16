import pandas as pd
from abc import abstractmethod, ABC


class DatasetSubmissionObject(ABC) : 

    def __init__(self, submission_df:pd.DataFrame) : 
        self.submission = submission_df


    @abstractmethod
    def get_text(self, **kwargs) : 
        pass

    @abstractmethod
    def get_groundtruth(self, **kwargs) : 
        pass

    @abstractmethod
    def get_predictions(self , **kwargs) : 
        pass