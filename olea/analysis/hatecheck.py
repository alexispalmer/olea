from typing import List, Union
import numpy as np

from olea.data.hatecheck import HateCheckSubmissionObject
from olea.utils.analysis_tools import get_metrics, get_examples
from olea.data.dso import DatasetSubmissionObject
from olea.analysis.generic import Generic

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
    def _run_analysis_on_functionality(cls, submission:HateCheckSubmissionObject, on:Union[str,List[str]],show_examples,plot) :
    
        if type(on) == str and on == "category":
            new_dic = {}
            for k,v in cls.categories.items():
                for x in v:
                    new_dic.setdefault(x,[]).append(k)
                    
            submission.submission[on] = submission.submission.apply (lambda row: new_dic[row.functionality][0], axis=1)
            return Generic.analyze_on(submission, on, show_examples, plot)
        
        
        if type(on) ==str and on in cls.categories : 
            #on is a str of one category
           df1 = submission.submission[submission.submission['functionality'].isin(cls.categories[on])]
           df2 = submission.submission[~submission.submission['functionality'].isin(cls.categories[on])]
           new_feature = on
           #analysis_set = submission.submission[submission.submission['functionality'].isin(cls.categories[on])]
           #find instances of feature vs not feture
           
        if set(on) <= cls.categories.keys():
            #On is a list of categories            
            categories = []
            for f in on:
                for x in cls.categories[f]:
                    categories.append(x)
            
            df1 = submission.submission[submission.submission['functionality'].isin(categories)]
            df2 = submission.submission[~submission.submission['functionality'].isin(categories)]
            new_feature = str(", ").join(on)
            
        labels = [new_feature, str("Not " + new_feature)]
        df1[new_feature] = labels[0]
        df2[new_feature] = labels[1]
        submission.submission = df1.merge(df2,"outer")
        return Generic.analyze_on (submission, new_feature, show_examples, plot)
           
           
        #elif type(on) == str : 
           # analysis_set = submission.submission[submission.submission['functionality'] == on]
        # else : 
        #     analysis_set = submission.submission[submission.submission['functionality'].isin(on)]

        
        #create new column
           

       
        # return get_metrics(df = analysis_set, 
        #                    off_col = cls.label_column, 
        #                    column = None)
    @classmethod
    def analyze_on(cls, hatecheck_submission:HateCheckSubmissionObject, on:Union[str, List[str]], show_examples = True, plot =True) : 
        return cls._run_analysis_on_functionality(hatecheck_submission, on, show_examples, plot)

    # @classmethod
    # def analyze_on_all(cls, hatecheck_submission:HateCheckSubmissionObject) :
    #     return cls._run_analysis_on_functionality(submission=hatecheck_submission.submission, on=cls.functionalities)


if __name__ == '__main__' : 

    import pandas as pd

    from olea.data.hatecheck import HateCheck

    hc = HateCheck()
    hc_data = hc.data()

    import numpy as np
    mock_preds = np.random.choice([0 , 1] , size=hc_data.shape[0])
    map = {1 : 'hateful', 0 : 'non-hateful'}

    hcso = hc.submit(hc_data, mock_preds, map)

    print('Analysis on acceptable abuse...')
    print(Generic.analyze_on(hcso, 'functionality'))

    print('Analysis on hateful abuse...')
    print(HateCheckAnalysis.analyze_on(hcso, ['profanity','threats','slurs']))

    print('Analysis on hateful abuse...')
    print(HateCheckAnalysis.analyze_on(hcso, 'category'))

    print('Analysis on negation...')
    print(HateCheckAnalysis.analyze_on(hcso, 'threats'))



    # print('Analysis on all functionalities...')
    # print(HateCheckAnalysis.analyze_on_all_functionalities(hcso))

    

    

    
    