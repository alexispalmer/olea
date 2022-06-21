import os
import numpy as np
import pandas as pd
import pickle
import requests

from src.data.dataset import Dataset
from src.data.dso import DatasetSubmissionObject


class COLD(Dataset) :

    def __init__(self, dataset_save_dir: str = 'datasets') -> None:
        super().__init__(dataset_save_dir)
        self.dataset_name = 'Complex and Offensive Language Dataset'
        self.URL = 'https://raw.githubusercontent.com/alexispalmer/cold-team/dev_main/data/cold_with_hx_preds.tsv?token=GHSAT0AAAAAABRAGOZYTN4XUQA6E7AXVASKYVDQEWA'
        self.BASEURL = os.path.basename(self.URL).split('?')[0]
        self.description = 'This is the dataset from COLD.'
        self.dataset_path = os.path.join(self.dataset_save_dir, self.BASEURL)
        self.data_columns = ['ID', 'DataSet', 'Text']
        self.label_columns = ['Off1', 'Off2', 'Off3']

    def _download(self) -> pd.DataFrame:
        r = requests.get(self.URL)
        lines = r.content.decode('utf-8').replace('\r' , '').split('\n')
        lines = [line.split('\t') for line in lines]

        header , data = lines[0] , lines[1:]

        data = [{h:d for h, d in zip(header, data_sample)} for data_sample in data ]
        df = pd.DataFrame(data)


        with open(self.dataset_path , 'wb') as f : 
            pickle.dump(df, f)
            
        return df

    def submit(self, dataset: pd.DataFrame, submission: iter, map: dict = None) -> DatasetSubmissionObject:
        submission_df = super().submit(dataset, submission, map)
        return COLDSubmissionObject(submission_df)


class COLDSubmissionObject(DatasetSubmissionObject) : 
    
    def __init__(self, submission_df: pd.DataFrame):
        super().__init__(submission_df)

    def filter_submission(self, on:str, filter:function, **kwargs):

        self.submission['filter_results'] = self.submission[on].apply(filter)
        filtered_submission = self.submission[self.submission['filter_results'] == True]

        if 'columns' in kwargs : 
            outputs = [filtered_submission[col] for col in kwargs['columns']]
            return outputs
    

        else : 
            return [filtered_submission['Text'] , filtered_submission['OffMaj'] , filtered_submission['preds']]



if __name__ == '__main__' : 

    cold = COLD('cold')
    cold._load_data()

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



 

