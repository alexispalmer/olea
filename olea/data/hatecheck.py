from olea.data.dataset import Dataset
from olea.data.dso import DatasetSubmissionObject

from datasets import load_dataset
import pandas as pd


class HateCheck(Dataset) : 

    def __init__(self, split='train') -> None:
        self.dataset_name = 'HateCheck'
        self.URL = 'COLD-team/HateCheck'
        self.description = 'This is the dataset for HateCheck.'
        
        self.features = ['functionality', 'case_id' , 'test_case' , 'direction' , 
                            'focus_words' , 'focus_lemma']
        self.text_column = 'test_case'
        self.gold_column ='label_gold'
        
        self.unique_labels = None
        self.split = 'test'

        self._data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        print('Loading data...')
        df = load_dataset(self.URL, 
                        use_auth_token='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')[self.split].to_pandas()

        df = df.drop(columns='Unnamed: 0')
        df['target_ident'] = df["target_ident"].astype(str)
        return df 

    def submit(self, dataset: pd.DataFrame, submission: iter, map: dict = None) -> DatasetSubmissionObject:
        return super().submit(dataset, submission, map)

if __name__ == '__main__' : 

    hc = HateCheck()
    hc_data = hc.data()

    print(pd.unique(hc._data[hc.label_columns].values.ravel('K')))

    import numpy as np
    mock_preds = np.random.choice([0 , 1] , size=hc_data.shape[0])
    map = {'hateful' : 1 , 'non-hateful' : 0}
    map = {1 : 'hateful', 0 : 'non-hateful'}

    hcso = hc.submit(hc_data, mock_preds, map)
    print(hcso.submission)



