from typing import Union, List

import pandas as pd
from sklearn.metrics import classification_report
import numpy as np

from src.data.cold import COLD, COLDSubmissionObject

class COLDAnalysis(object) : 

    label_column = 'Off'
    prediction_column = 'preds'

    slur_column = 'Slur'
    nom_column = 'Nom'
    dist_column = 'Dist'
    category_column = 'Cat'

    offensive_columns = ['Off1' , 'Off2' , 'Off3']
    slurs_columns = ['Slur1' , 'Slur2' , 'Slur3']
    nom_columns = ['Nom1' , 'Nom2' , 'Nom3']
    slurs_columns = ['Slur1' , 'Slur2' , 'Slur3']
    dist_columns = ['Dist1' , 'Dist2' , 'Dist3']

    @classmethod
    def _run_analysis_on(cls, submission:pd.DataFrame, 
                        on:Union[str, List[str]], 
                        target_column:str='Off', 
                        consider_only_true_targets:bool=False
                        ) : 

        if type(on) == str : 
            analysis_set = submission[submission[on] == 'Y']

        else : 
            analysis_set = submission
            for o in on : analysis_set = analysis_set[analysis_set[o] == 'Y']
                
        if consider_only_true_targets : 
                analysis_set = analysis_set[analysis_set[target_column] == 'Y']

        return classification_report(analysis_set[target_column] , analysis_set[cls.prediction_column])

    @classmethod
    def analyze_on(cls, cold_submission:COLDSubmissionObject, 
                    features:Union[str, List[str]], 
                    only_offensive_examples:bool=False) : 

        return cls._run_analysis_on(cold_submission.submission, 
                                    on=features, 
                                    target_column=cls.label_column, 
                                    consider_only_true_targets=only_offensive_examples)


        
        


if __name__ == '__main__'  : 

    cold = COLD()
    dataset = cold._data
    num_preds = dataset.shape[0]
    bool_preds = np.random.choice([True, False], size=num_preds)
    map = {True : 'Y' , False : 'N'}
    cso = cold.submit(dataset, bool_preds, map=map)

    print(COLDAnalysis.analyze_on(cso, ['Nom' , 'Slur']))
    print(cso)
    print(COLDAnalysis.analyze_on(cso, 'Dist'))
















