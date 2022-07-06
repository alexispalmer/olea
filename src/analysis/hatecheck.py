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

    @classmethod
    def _run_analysis_on_functionality(cls, submission:HateCheckSubmissionObject, on:Union[str,List[str]]) :
        
        if type(on) == str : 
            analysis_set = submission[submission['functionality'] == on]
        else : 
            analysis_set = submission[submission['functionality'].isin(on)]

        return classification_report(analysis_set[cls.label_column] , 
                                    analysis_set[cls.prediction_column], 
                                    zero_division=0)
    @classmethod
    def analyze_on_derogation(cls, hatecheck_submission:HateCheckSubmissionObject) : 
        derogation_columns = ['derog_neg_emote_h', 'derog_neg_attrib_h', 'derog_dehum_h', 'derog_impl_h']
        return cls._run_analysis_on_functionality(submission=hatecheck_submission.submission, on=derogation_columns)

    @classmethod
    def analyze_on_threats(cls, hatecheck_submission:HateCheckSubmissionObject) : 
        threat_columns = ['threat_dir_h', 'threat_norm_h']
        return cls._run_analysis_on_functionality(submission=hatecheck_submission.submission, on=threat_columns)

    @classmethod
    def analyze_on_slurs(cls, hatecheck_submission:HateCheckSubmissionObject) : 
        slur_columns = ['slur_h', 'slur_homonym_nh', 'slur_reclaimed_nh']
        return cls._run_analysis_on_functionality(submission=hatecheck_submission.submission, on=slur_columns)

    @classmethod
    def analyze_on_profanity(cls, hatecheck_submission:HateCheckSubmissionObject) : 
        profanity_columns = ['profanity_h', 'profanity_nh']
        return cls._run_analysis_on_functionality(submission=hatecheck_submission.submission, on=profanity_columns)

    @classmethod
    def analyze_on_pronoun_reference(cls, hatecheck_submission:HateCheckSubmissionObject) : 
        pronoun_reference_columns = ['ref_subs_clause_h','ref_subs_sent_h']
        return cls._run_analysis_on_functionality(submission=hatecheck_submission.submission, on=pronoun_reference_columns)

    @classmethod
    def analyze_on_negation(cls, hatecheck_submission:HateCheckSubmissionObject) : 
        negation_columns = ['ref_subs_clause_h','ref_subs_sent_h']
        return cls._run_analysis_on_functionality(submission=hatecheck_submission.submission, on=negation_columns)

    @classmethod
    def analyze_on_phrasing(cls, hatecheck_submission:HateCheckSubmissionObject) : 
        phrasing_columns = ['phrase_question_h','phrase_opinion_h']
        return cls._run_analysis_on_functionality(submission=hatecheck_submission.submission, on=phrasing_columns)

    @classmethod 
    def analyze_on_non_hate(cls, hatecheck_submission:HateCheckSubmissionObject) : 
        non_hate_columns = ['ident_neutral_nh', 'ident_pos_nh']
        return cls._run_analysis_on_functionality(submission=hatecheck_submission.submission, on=non_hate_columns)

    @classmethod 
    def analyze_on_counter_speech(cls, hatecheck_submission:HateCheckSubmissionObject) : 
        counter_columns = ['counter_quote_nh','counter_ref_nh']
        return cls._run_analysis_on_functionality(submission=hatecheck_submission.submission, 
                                                on=counter_columns)

    @classmethod 
    def analyze_on_acceptable_abuse(cls, hatecheck_submission:HateCheckSubmissionObject) : 
        acceptable_abuse_columns = ['target_obj_nh', 'target_indiv_nh', 'target_group_nh']
        return cls._run_analysis_on_functionality(submission=hatecheck_submission.submission, 
                                                on=acceptable_abuse_columns)

    @classmethod 
    def analyze_on_hateful_abuse(cls, hatecheck_submission:HateCheckSubmissionObject) : 
        hateful_abuse_columns = ['spell_char_swap_h', 'spell_char_del_h', 'spell_space_del_h',
                                'spell_space_add_h', 'spell_leet_h']
        return cls._run_analysis_on_functionality(submission=hatecheck_submission.submission, 
                                                on=hateful_abuse_columns)


    @classmethod
    def analyze_on_all_functionalities(cls, hatecheck_submission:HateCheckSubmissionObject) : 
        return cls._run_analysis_on_functionality(submission=hatecheck_submission.submission, 
                                                on=cls.functionalities)




        
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
    print(HateCheckAnalysis.analyze_on_acceptable_abuse(hcso))

    print('Analysis on hateful abuse...')
    print(HateCheckAnalysis.analyze_on_hateful_abuse(hcso))

    print('Analysis on negation...')
    print(HateCheckAnalysis.analyze_on_negation(hcso))

    print('Analysis on all functionalities...')
    print(HateCheckAnalysis.analyze_on_all_functionalities(hcso))

    

    

    
    