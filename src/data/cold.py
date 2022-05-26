from src.data.dataset import Dataset

class COLD(Dataset) :

    def __init__(self, dataset_save_dir: str = '~/cold/') -> None:
        super().__init__(dataset_save_dir)
        self.dataset_name = 'Complex and Offensive Language Dataset'
        self.URL = 'https://raw.githubusercontent.com/alexispalmer/cold/master/data/cold-2016-majVote-fineGrained-needs-cleanup.tsv'


    
