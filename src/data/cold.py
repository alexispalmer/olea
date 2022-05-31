import os 
import requests
import pickle

import pandas as pd

from src.data.dataset import Dataset

class COLD(Dataset) :

    def __init__(self, dataset_save_dir: str = './cold/') -> None:
        super().__init__(dataset_save_dir)
        self.dataset_name = 'Complex and Offensive Language Dataset'
        self.URL = 'https://raw.githubusercontent.com/alexispalmer/cold/master/data/cold_mock_data.tsv'
        self.BASEURL = os.path.basename(self.URL)
        self.description = 'This is the dataset from COLD.'
        self.dataset_path = os.path.join(self.dataset_save_dir, self.BASEURL)
        self.i = 0

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
        return self._data

    def generator(self, batch_size) -> None:
        
        while True : 

            start = self.i 
            end = self.i + batch_size

            if end > self._data.shape[0] : 
                start = self.i 
                end = self._data.shape[0]

            elif end == self._data.shape[0] : 
                self.i = 0 
                start = self.i
                end = self.i + batch_size

            yield self._data.iloc[start:end, :]

    

        
        
if __name__ == '__main__' : 

    cold = COLD()
    cold._load_data()
    print(cold.data())
    # print(len(cold.data))
    # print(cold.data.head())
    # cold.data[cold.data['ID'=='D-373']]
    
