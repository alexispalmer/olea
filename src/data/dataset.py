
class Dataset(object) : 

    def __init__(self, 
                dataset_save_dir:str='~/cold/') -> None:
        """Initializes the dataset instance. 

        The requested dataset is downloaded and saved on the local system. You can choose where the dataset 
        should be saved by providing a dataset_save_dir.

        Args:
            dataset_save_dir (str, optional): _description_. Defaults to '~/cold/'.
        """
        
        self.COLD_URL = 'https://raw.githubusercontent.com/alexispalmer/cold/master/data/'
        self.dataset_save_dir = dataset_save_dir
        self.dataset_name = None
        self.shape = None
        self.description = None
        pass

    def __call__(self) -> None:
        pass

    def __iter__(self) -> None: 
        pass

    def __str__(self) -> str:
        pass


    def _download(self, dataset_name:str) -> bool : 
        """Base class that downloads the dataset from an online path. 

        Args:
            dataset_name (str): Name of the dataset from the repo

        Returns:
            bool: _description_
        """
        pass

    def data(self) : 
        pass

    def generator(self) -> None : 
        pass

    def info(self) -> None : 
        pass

    def analyze(self) -> None : 
        pass

