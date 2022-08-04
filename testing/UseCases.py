'Testing different HuggingFace models on the datasets'


from olea.data.cold import COLD

from olea.analysis.cold import COLDAnalysis
from olea.analysis.generic import Generic
from olea.analysis.hatecheck import HateCheckAnalysis
from olea.data.hatecheck import HateCheck
from olea.utils import preprocess_text 

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TextClassificationPipeline

MODELS = {"HateXplain_COLD": {"link" : "Hate-speech-CNERG/bert-base-uncased-hatexplain",
                              "map" : {"offensive": 'Y' , "hate speech": 'Y' , "normal":'N'}
                              },
          "HateXplain_HC": {"link" : "Hate-speech-CNERG/bert-base-uncased-hatexplain",
                            "map" : {"offensive": 'hateful' , "hate speech": 'hateful' , "normal":'non-hateful'}
                            },
          "Random_COLD":{"link" : None,
                         "map" : {True : 'Y' , False:'N'}
                         },
          "Random_HC":  {"link" : None,
                         "map" : {True : 'hateful', False : 'non-hateful'}
                         },
          "Roberta_COLD": {"link" : "cardiffnlp/twitter-roberta-base-offensive",
                         "map" : {"LABEL_0": 'N', 'LABEL_1': "Y"}
                         },
          
          "Roberta_HC": {"link" : "cardiffnlp/twitter-roberta-base-offensive",
                         "map" :{"LABEL_0": 'non-hateful', 'LABEL_1': "hateful"}
                         }
          
          }

def get_submission (model_name: str, dataset_name :str):
    #Load in COLD
    
    if dataset_name == "COLD":
        dataset = COLD()
        text_label = "Text"
        
    elif dataset_name == "Hatecheck":
        dataset= HateCheck()
        text_label = "test_case"
        
    pt = preprocess_text.PreprocessText()
    processed_text = pt.execute(dataset.data()[text_label])
    dataset.data()[text_label] =processed_text
    
          
    
    if model_name == "Random_HC" or model_name == "Random_COLD":
        num_preds = dataset.data().shape[0]
        preds = np.random.choice([True, False], size=num_preds)
    else:
        #load in model infomation
        tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name]["link"])
        model = AutoModelForSequenceClassification.from_pretrained(MODELS[model_name]["link"])
        #Create Pipeline for Predicting
        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
        predicted = pd.DataFrame(pipe(list(dataset.data()[text_label])))
        preds = predicted.label

    #create submission object
    if dataset_name == "COLD":
        submission = dataset.submit(dataset.data(), preds, map=MODELS[model_name]["map"])
    elif dataset_name == "Hatecheck":
        submission = dataset.submit(dataset.data(), preds, map=MODELS[model_name]["map"])
    
    return submission

def run_analysis_generic(submission):
    results = {}
    results["aave"] = Generic.aave(submission,show_examples = True)
    results["female"] = Generic.check_substring(submission,'female',show_examples = True)
    results["str_len"] = Generic.str_len_analysis(submission,show_examples = True)
    return results

def run_analysis_COLD(submission):
    results = {}
    results["cold_cat"] = COLDAnalysis.analyze_on(submission,'Cat',show_examples = True)
    results["coarse"] = COLDAnalysis.analyze_on(submission, 'Off',show_examples = True)
    results["anno_agree"] = Generic.check_anno_agreement(submission, ["Off1","Off2","Off3"],show_examples = True)
    return results

def run_analysis_HC(submission):
    results = {}
    results["target_ident"] = Generic.analyze_on(submission,'target_ident')
    results["category"] = HateCheckAnalysis.analyze_on(submission, 'category')
    return results
    
if __name__ == '__main__' : 
    # hcso = get_submission("HateXplain_HC", "Hatecheck")
    #results_hc_g = run_analysis_generic(hcso)
    # results_hc = run_analysis_HC(hcso)
    # coldso = get_submission("HateXplain_COLD", "COLD")
    # results_cold_g= run_analysis_generic(coldso)
    # results_cold = run_analysis_COLD(coldso)
    
    ##LATEX EXAMPLES
    #rso_cold= get_submission("Roberta_COLD", "COLD")
    # results_cold_g= run_analysis_generic(rso_cold)
    # results_cold = run_analysis_COLD(rso_cold)
    
    rso_hc = get_submission("Roberta_HC", "Hatecheck")
    # results_hc = run_analysis_HC(rso_hc)
    
    
    ##
    # submission = get_submission("Random_HC",'Hatecheck')
    # results = run_analysis_generic(submission)
