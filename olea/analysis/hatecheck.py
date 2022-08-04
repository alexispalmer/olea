from typing import List, Union
import numpy as np

from olea.utils.analysis_tools import get_metrics, get_examples
from olea.data.dso import DatasetSubmissionObject
from olea.analysis.generic import Generic

class HateCheckAnalysis(object) : 

    gold_column = 'label_gold'
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
                'spelling changes' : ['spell_char_swap_h', 'spell_char_del_h', 'spell_space_del_h',
                                'spell_space_add_h', 'spell_leet_h']
                }
    
    category_hate_label = {'derogation' : '(h)', 
                'threats' : '(h)' ,
                'slurs': "",
                'profanity' : "", 
                'pronoun_references': '(h)', 
                'negation':"", 
                'phrasing' : '(h)', 
                'identity' : '(nh)' ,
                'counter' : '(nh)',
                'nonhateful-abuse' : '(nh)',
                'spelling changes' :  '(h)'
                }

    @classmethod
    def _run_analysis_on_functionality(cls, submission:DatasetSubmissionObject, on:Union[str,List[str]],show_examples,plot,savePlotToFile) :
        """helper function for running analysis on a category, a list of categories, or over all categories. Returns two dataframes. plot_info corresponds to 
            information that is plotted, number of offensive/non offensive instances for each category in "on" as well as
            accuracy of model. Metrics returns the classification report for each category specified on "on"

        Args:
            submission (HateCheckSubmissionObject): submission object to run analysis on
            on (str or List of str): what to run analysis on : "category" for all categories, a specific category name as defined in 
                class (eg: 'threats'), or list of categories (eg: ['threats','slurs'])
            plot (boolean): to plot results or not
            show_examples (boolean): to return examples or not
            savePlotToFile (str): File name for saving plot, empty string will not save a plot

        Returns:
            plot_info (df) : results corresponding to plotted information:(total instances for each category,
                             total correctly predicted, and accuracy)
            metrics (df): classification report for each category
        """
        
        #create the categories as defined in class - use reverse dictionary lookup to assign fine graiend labels to broader categories
        if type(on) == str and on == "category":
            new_dic = {}
            for k,v in cls.categories.items():
                for x in v:
                    new_dic.setdefault(x,[]).append(k)
                    
            submission.submission[on] = submission.submission.apply (lambda row: new_dic[row.functionality][0], axis=1)
            return Generic.analyze_on(submission, on, show_examples, plot,savePlotToFile)
        
        
        # analyze on one of the categories
        if type(on) ==str and on in cls.categories : 
            #on is a str of one category
           df1 = submission.submission[submission.submission['functionality'].isin(cls.categories[on])]
           df2 = submission.submission[~submission.submission['functionality'].isin(cls.categories[on])]
           new_feature = on
           #analysis_set = submission.submission[submission.submission['functionality'].isin(cls.categories[on])]
           #find instances of feature vs not feture
           
        #Analyze on a list of categories    
        if set(on) <= cls.categories.keys():       
            categories = []
            for f in on:
                for x in cls.categories[f]:
                    categories.append(x)
            
            df1 = submission.submission.loc[submission.submission['functionality'].isin(categories)].copy()
            df2 = submission.submission.loc[~submission.submission['functionality'].isin(categories)].copy()
            new_feature = str(", ").join(on)
            
        labels = [new_feature, str("Not " + new_feature)]
        df1[new_feature] = labels[0]
        df2[new_feature] = labels[1]
        submission.submission = df1.merge(df2,"outer")
        return Generic.analyze_on (submission, new_feature, show_examples, plot,savePlotToFile)
           
           
    @classmethod
    def analyze_on(cls, hatecheck_submission:DatasetSubmissionObject, on:Union[str, List[str]], show_examples = True, plot =True,savePlotToFile = "") : 
        """function for running analysis on a category, a list of categories, or over all categories. Returns two dataframes. plot_info corresponds to 
            information that is plotted, number of offensive/non offensive instances for each category in "on" as well as
            accuracy of model. Metrics returns the classification report for each category specified on "on"

        Args:
            submission (COLDSubmissionObject): submission object to run analysis on
            on (str or List of str): what to run analysis on : "category" for all categories, a specific category name as defined in 
                class (eg: 'threats'), or list of categories (eg: ['threats','slurs'])
            plot (boolean): to plot results or not
            show_examples (boolean): to return examples or not
            savePlotToFile (str): File name for saving plot, empty string will not save a plot

        Returns:
            plot_info (df) : results corresponding to plotted information:(total instances for each category,
                             total correctly predicted, and accuracy)
            metrics (df): classification report for each category
        """
        return cls._run_analysis_on_functionality(hatecheck_submission, on, show_examples, plot,savePlotToFile)

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

    

    

    
    