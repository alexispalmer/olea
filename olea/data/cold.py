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
        
        self.features = ['ID', 'DataSet', 'Text']
        self.label_columns = ['Off', 'Slur', 'Nom', 'Dist']
        self.gold_column = 'Off'
        self.text_column = 'Text'
        
        self.unique_labels = None
        self.split = split

        self._data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        print('Loading data...')
        return load_dataset(self.URL)[self.split].to_pandas()

    def submit(self, dataset: pd.DataFrame, submission: iter, map: dict = None) -> DatasetSubmissionObject:
        return super().submit(dataset, submission, map)




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



 

