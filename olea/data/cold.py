import numpy as np
import pandas as pd
from olea.data.dataset import Dataset
from olea.data.dso import DatasetSubmissionObject
from datasets import load_dataset

class COLD(Dataset) :

    def __init__(self, split='train') -> None:
        self.dataset_name = 'Complex and Offensive Language Dataset'
        self.URL = 'COLD-team/COLD'
        self.description = 'This is the dataset from COLD.'
        self.data_columns = ['ID', 'DataSet', 'Text']
        self.label_columns = ['Off', 'Slur', 'Nom', 'Dist']
        self.label_column = 'Off'
        self.unique_labels = None
        self.split = split

        self._data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        print('Loading data...')
        return load_dataset(self.URL,sep='\t')[self.split].to_pandas()

    def submit(self, dataset: pd.DataFrame, submission: iter, map: dict = None) -> DatasetSubmissionObject:
        submission_df = super().submit(dataset, submission, map)
        return COLDSubmissionObject(submission_df)


class COLDSubmissionObject(DatasetSubmissionObject) : 

    label_column = 'Off'
    prediction_column = 'preds'
    text_column = 'Text'
    data_columns = ['Text' , 'Off' , 'Nom' , 'Slur' , 'Dist']
    
    def __init__(self, submission: DatasetSubmissionObject):
        self.submission = submission.submission

    def filter_submission(self, on:str, filter:callable, **kwargs):

        self.submission['filter_results'] = self.submission[on].apply(filter)
        filtered_submission = self.submission[self.submission['filter_results'] == True]

        if 'columns' in kwargs : 
            outputs = [filtered_submission[col] for col in kwargs['columns']]
            return outputs
    
        else : 
            return [filtered_submission['Text'] , filtered_submission['Off'] , filtered_submission['preds']]



if __name__ == '__main__' : 

    cold = COLD()


    print('Testing Generator')

    gen = cold.generator(64)    


    print(next(gen))
    print(next(gen))
    print(next(gen))

    print('Testing submission')

    dataset = next(gen)

    dataset.head()

    num_preds = dataset.shape[0]

    yn_preds = np.random.choice(['Y' , 'N'], size=num_preds)
    bool_preds = np.random.choice([True, False], size=num_preds)

    map = {True : 'Y' , False:'N'}

    print('Yes-No Preds')

    analysis = cold.submit(dataset, yn_preds)



 

