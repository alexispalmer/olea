import pandas as pd

from olea.data.dataset import Dataset
from olea.data.dso import DatasetSubmissionObject

from datasets import load_dataset

class HateXplain(Dataset) : 

    def __init__(self, split='train') -> None:
        self.dataset_name = 'HateXplain'
        self.URL = 'hatexplain'
        self.description = 'This is the dataset for HateXplain.'
        
        self.features = ['ID', 'post_tokens']
        self.gold_column = ['label1' , 'label2' , 'label3']
        self.text_column = 'post_tokens'
       
        self.unique_labels = None
        self.split = split

        self._data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        print('Loading data...')
        df = load_dataset(self.URL)[self.split].to_pandas()

        labels = []
        annotator_ids = []
        targets = []
        rationales = []
        
        for i , row in df.iterrows() : 
            labels.append(row.annotators['label'])
            annotator_ids.append(row.annotators['annotator_id'])
            targets.append(row.annotators['target'])

        df[['label1' , 'label2' , 'label3']] = labels
        df[['anno_id1' , 'anno_id2' , 'anno_id3']] = annotator_ids
        df[['target1' , 'target2' , 'target3']] = targets

        return df
            



    def submit(self, dataset: pd.DataFrame, submission: iter, map: dict = None) -> DatasetSubmissionObject:
        return super().submit(dataset, submission, map)

