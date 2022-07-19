import pandas as pd 
import numpy as np 


from src.data.dataset import Dataset

olid = pd.read_csv('data/olid/levela-combined.csv')

olid_dataset = Dataset(olid, data_columns='Text' , label_column = 'label', text_column='Text')


data_gen = olid_dataset.generator(batch_size=64)

data = next(data_gen)
data1 = next(data_gen)

preds = np.random.choice([True, False] , size = len(data))

submission = olid_dataset.submit(data, preds, {True : 'OFF' , False : 'NOT'})

from src.analysis.generic import Generic

print(Generic.analyze_on(submission, features = 'sanity_check', plot=False, show_examples=True))

# What is Y what is N? 
# print(Generic.aave(submission))

# print(Generic.str_len_analysis(submission))

# print(Generic.check_substring(submission, "MAGA"))


