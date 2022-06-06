import os
import numpy as np
import pandas as pd
import pickle
import requests
from src.data.dataset import Dataset

class COLD(Dataset) :

    def __init__(self, dataset_save_dir: str = '~/cold/') -> None:
        super().__init__(dataset_save_dir)
        self.dataset_name = 'Complex and Offensive Language Dataset'
        self.URL = 'https://raw.githubusercontent.com/alexispalmer/cold-team/dj_dev/data/cold_mock_data.tsv?token=GHSAT0AAAAAABUE7TWTU4XK5XCKFNN54OQ4YUZGUYQ'
        self.BASEURL = os.path.basename(self.URL)
        self.description = 'This is the dataset from COLD.'
        self.dataset_path = os.path.join(self.dataset_save_dir, self.BASEURL)
        self.data_columns = ['ID', 'DataSet', 'Text']
        self.label_columns = ['Off1', 'Off2', 'Off3']

    def _download(self) -> None :
        r = requests.get(self.URL)
        lines = r.content.decode('utf-8').replace('\r' , '').split('\n')
        lines = [line.split('\t') for line in lines]

        header , data = lines[0] , lines[1:]

        data = [{h:d for h, d in zip(header, data_sample)} for data_sample in data ]
        df = pd.DataFrame(data)


        with open(self.dataset_path , 'wb') as f : 
            pickle.dump(df, f)
            
        return df


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

    cold.submit(dataset, yn_preds)
    
    print('True-False Preds')

    cold.submit(dataset, bool_preds , map)


 

