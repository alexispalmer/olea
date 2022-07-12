from re import T
import string
import random
import pandas as pd 

keys = "ID	DataSet	Text	Off1	Off2	Off3	Slur1	Slur2	Slur3	Nom1	Nom2	Nom3	Dist1	Dist2	Dist3"



def generate_id_dataset() : 
    char = random.choices(string.ascii_uppercase, k=1)[0]
    num = ''.join(random.choices(string.digits, k=random.randint(3,4)))
    return [char + '-' + num , char]

def generate_text() : 
    
    text = []

    for i in range(random.randint(5, 20)) : 
        word = ''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(1,10)))
        text.append(word)

    return [' '.join(text)]

def generate_option() :

    return random.choices('NY' , k=12)

mock_data = []
mock_data.append(keys)
    
for i in range(100) : 

    id_dataset = generate_id_dataset()
    text = generate_text()
    options = generate_option()

    line = '\t'.join(id_dataset + text + options)
    mock_data.append(line)


mock_data = '\n'.join(mock_data)

with open('data/cold_mock_data.tsv' , 'w') as f : 
    f.write(mock_data)

df = pd.read_csv('data/cold_mock_data.tsv', sep='\t')

# with open('cold/cold_mock_data.tsv' , 'wb') as f : 
#     pickle.dump(df , f)