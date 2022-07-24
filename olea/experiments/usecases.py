import pandas as pd 
import numpy as np
from olea.analysis.generic import Generic 


from olea.data.dataset import Dataset
from olea.data.dso import DatasetSubmissionObject

olid = pd.read_csv('data/olid/levela-combined.csv')

olid_dataset = Dataset(olid, data_columns='Text' , label_column = 'label', text_column='Text')


data_gen = olid_dataset.generator(batch_size=64)

data = next(data_gen)
data1 = next(data_gen)

preds = np.random.choice([True, False] , size = len(data))

submission = olid_dataset.submit(data, preds, {True : 'OFF' , False : 'NOT'})

# from src.analysis.generic import Generic

# print(Generic.analyze_on(submission, features = 'sanity_check', plot=False, show_examples=True))

# What is Y what is N? 
# print(Generic.aave(submission))

# print(Generic.str_len_analysis(submission))

# print(Generic.check_substring(submission, "MAGA"))

from olea.data.dataset import Dataset
from olea.utils.analysis_tools import get_metrics, get_examples

class OLIDDataset(Dataset) : 

    text_column = 'Text'
    label_column = 'label'
    data_columns = ['Text' , 'label_id']

    def __init__(self, olid_csv_path:str) -> None:
        self.olid_csv_path = olid_csv_path
        self._data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.olid_csv_path)




class OLIDAnalysis(object) : 

    @classmethod
    def analyse_on(cls, submission:DatasetSubmissionObject, on:str) : 
        '''
        Unique OLID analysis goes here!
        '''
        return get_metrics(submission, on)



olid = OLIDDataset('data/olid.csv')

datagen = olid.generator(64)
data = next(datagen)
preds = model.predict(data)
map = {'OFF' : 1.0 , 'NOT' : 0.0}

oso = olid.submit(data, preds, map)

OLIDAnalysis.analyse_on(oso, 'label')
Generic.check_substring(submission, 'female')





        


