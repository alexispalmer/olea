import os
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
        self.i = 0
        self.data_columns = ['ID', 'DataSet', 'Text']

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

    def data(self) : 
        '''
        TODO : Filters to apply before getting the data. 
        '''
        return self._data[self.data_columns]

    def generator(self, batch_size) -> None:
        
        start = 0
        end = batch_size

        while True : 

            if end < self._data.shape[0] :

                batch = self._data.loc[start:end, self.data_columns]
                start +=  batch_size
                end  = start + batch_size

            elif end >= self._data.shape[0] : 
                start, end = self._data.shape[0] - batch_size, self._data.shape[0]
                batch = self._data.loc[start:end, self.data_columns]
                start, end = 0, batch_size
                

            yield batch

    

if __name__ == '__main__' : 

    cold = COLD('cold')
    cold._load_data()

    gen = cold.generator(64)    

    print(next(gen))
    print(next(gen))
    print(next(gen))
 

