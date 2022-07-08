from typing import List, Union
import numpy as np
from sklearn.metrics import classification_report


from src.data.hatecheck import HateCheckSubmissionObject
# from src.utils.analysis_tools import get_metrics, get_examples
from src.data.dso import DatasetSubmissionObject

class HateCheckAnalysis(object) : 

    label_column = 'label_gold'
    prediction_column = 'preds'

    data_columns = ['functionality', 'case_id' , 'test_case' , 'direction' , 
                    'focus_words' , 'focus_lemma']

    functionalities = ['derog_neg_emote_h', 'derog_neg_attrib_h', 'derog_dehum_h', 'derog_impl_h',
                    'threat_dir_h', 'threat_norm_h', 'slur_h', 'slur_homonym_nh',
                    'slur_reclaimed_nh', 'profanity_h', 'profanity_nh', 'ref_subs_clause_h',
                    'ref_subs_sent_h', 'negate_pos_h', 'negate_neg_nh', 'phrase_question_h',
                    'phrase_opinion_h', 'ident_neutral_nh', 'ident_pos_nh', 'counter_quote_nh',
                    'counter_ref_nh', 'target_obj_nh', 'target_indiv_nh', 'target_group_nh',
                    'spell_char_swap_h', 'spell_char_del_h', 'spell_space_del_h',
                    'spell_space_add_h', 'spell_leet_h']

    categories = {'derogation' : ['derog_neg_emote_h', 'derog_neg_attrib_h', 'derog_dehum_h', 'derog_impl_h'], 
                'threats' : ['threat_dir_h', 'threat_norm_h'],
                'slurs':['slur_h', 'slur_homonym_nh', 'slur_reclaimed_nh'],
                'profanity' : ['profanity_h', 'profanity_nh'], 
                'pronoun_references':['ref_subs_clause_h','ref_subs_sent_h'], 
                'negation':['negate_pos_h', 'negate_neg_nh'], 
                'phrasing' : ['phrase_question_h','phrase_opinion_h'], 
                'identity' : ['ident_neutral_nh', 'ident_pos_nh'],
                'counter' : ['counter_quote_nh','counter_ref_nh'],
                'nonhateful-abuse' : ['target_obj_nh', 'target_indiv_nh', 'target_group_nh'],
                'hateful-abuse' : ['spell_char_swap_h', 'spell_char_del_h', 'spell_space_del_h',
                                'spell_space_add_h', 'spell_leet_h']
                }

    @classmethod
    def _run_analysis_on_functionality(cls, submission:HateCheckSubmissionObject, on:Union[str,List[str]]) :
        
        if on in cls.categories : 
            analysis_set = submission[submission['functionality'].isin(cls.categories[on])]
        elif type(on) == str : 
            analysis_set = submission[submission['functionality'] == on]
        else : 
            analysis_set = submission[submission['functionality'].isin(on)]

        return classification_report(analysis_set[cls.label_column] , 
                                    analysis_set[cls.prediction_column], 
                                    zero_division=0)
    @classmethod
    def analyze_on(cls, hatecheck_submission:HateCheckSubmissionObject, on:Union[str, List[str]]) : 
        return cls._run_analysis_on_functionality(submission=hatecheck_submission.submission, on=on)

    @classmethod
    def analyze_on_all(cls, hatecheck_submission:HateCheckSubmissionObject) :
        return cls._run_analysis_on_functionality(submission=hatecheck_submission.submission, on=cls.functionalities)


if __name__ == '__main__' : 

    import pandas as pd

    from src.data.hatecheck import HateCheck

    hc = HateCheck()
    hc_data = hc.data()

    import numpy as np
    mock_preds = np.random.choice([0 , 1] , size=hc_data.shape[0])
    map = {1 : 'hateful', 0 : 'non-hateful'}

    hcso = hc.submit(hc_data, mock_preds, map)

    print('Analysis on acceptable abuse...')
    print(HateCheckAnalysis.analyze_on(hcso, 'nonhateful-abuse'))

    print('Analysis on hateful abuse...')
    print(HateCheckAnalysis.analyze_on(hcso, 'hateful-abuse'))

    print('Analysis on negation...')
    print(HateCheckAnalysis.analyze_on(hcso, 'negation'))

    print('Analysis on all functionalities...')
    print(HateCheckAnalysis.analyze_on_all_functionalities(hcso))

    

    

    
    