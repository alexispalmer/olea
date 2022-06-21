import pandas as pd
from abc import abstractmethod, ABC

from typing import List, Tuple

class DatasetSubmissionObject(ABC) : 

    def __init__(self, submission_df:pd.DataFrame) : 
        self.submission = submission_df


    @abstractmethod
    def filter_submission(self, on:str, filter:function, **kwargs)  : 
        pass
