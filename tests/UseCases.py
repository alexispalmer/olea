'Testing different HuggingFace models on the datasets'

from src.data.cold import COLD, COLDSubmissionObject
from src.analysis.cold import COLDAnalysis
from src.analysis.generic import Generic
from src.data.hatecheck import HateCheck

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
        cold = COLD()
        dataset = cold._data
        text_label = "Text"
    elif dataset_name == "Hatecheck":
        hc = HateCheck()
        dataset = hc.data()
        text_label = "test_case"
    
    if model_name == "Random_HC" or model_name == "Random_COLD":
        num_preds = dataset.shape[0]
        preds = np.random.choice([True, False], size=num_preds)
    else:
        #load in model infomation
        tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name]["link"])
        model = AutoModelForSequenceClassification.from_pretrained(MODELS[model_name]["link"])
        #Create Pipeline for Predicting
        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
        predicted = pd.DataFrame(pipe(list(dataset[text_label])))
        preds = predicted.label

    #create submission object
    if dataset_name == "COLD":
        submission = cold.submit(cold.data(), preds, map=MODELS[model_name]["map"])
    elif dataset_name == "Hatecheck":
        submission = hc.submit(dataset, preds, map=MODELS[model_name]["map"])
    
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
    return results
    
if __name__ == '__main__' : 
    # hcso = get_submission("HateXplain_HC", "Hatecheck")
    # results_hc_g = run_analysis_generic(hcso)
    # results_hc = run_analysis_HC(hcso)
    
    # coldso = get_submission("HateXplain_COLD", "COLD")
    # results_cold_g= run_analysis_generic(coldso)
    # results_cold = run_analysis_COLD(coldso)
    
    rso_hc = get_submission("Roberta_HC", "Hatecheck")
    results_hc_g = run_analysis_generic(rso_hc)
    results_hc = run_analysis_HC(rso_hc)
    
    rso_cold= get_submission("Roberta_COLD", "COLD")
    results_cold_g= run_analysis_generic(rso_cold)
    results_cold = run_analysis_COLD(rso_cold)