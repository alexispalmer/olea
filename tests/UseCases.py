'Testing different HuggingFace models on the datasets'

from src.data.cold import COLD, COLDSubmissionObject
from src.analysis.cold import COLDAnalysis
from src.analysis.generic import Generic

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TextClassificationPipeline

MODELS = {"HateXplain": {"link" : "Hate-speech-CNERG/bert-base-uncased-hatexplain",
                         "map" : {"offensive": 'Y' , "hate speech": 'Y' , "normal":'N'}
                         },
          "KcELECTRA" : {"link" : "beomi/beep-KcELECTRA-base-hate",
                         "map" : {"offensive": 'Y' , "hate": 'Y' , "none":'N'}
                         },
          "Random":     {"link" : None,
                         "map" : {True : 'Y' , False:'N'}
                         }
          }

def get_submission (model_name: str):
    #Load in COLD [IMPORTANT TO DROP FINAL ROW (IT"S NAN)]
    cold = COLD()
    dataset = cold._data
    
    if model_name == "Random":
        num_preds = dataset.shape[0]
        preds = np.random.choice([True, False], size=num_preds)
    else:
        #load in model infomation
        tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name]["link"])
        model = AutoModelForSequenceClassification.from_pretrained(MODELS[model_name]["link"])
        #Create Pipeline for Predicting
        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
        predicted = pd.DataFrame(pipe(list(dataset["Text"])))
        preds = predicted.label

    #create submission object
    submission = cold.submit(cold.data(), preds, map=MODELS[model_name]["map"])
    return submission

def run_analysis (submission):
    results = {}
    results["cold_cat"] = COLDAnalysis.analyze_on(submission,'Cat')
    results["anno_agree"] = Generic.check_anno_agreement(submission, ["Off1","Off2","Off3"],off_col="Off",show_examples = True)
    results["coarse"] = COLDAnalysis.analyze_on(submission, 'Off')
    results["aave"] = Generic.aave(submission)
    results["#"] = Generic.check_substring("#",submission)
    results["str_len"] = Generic.str_len_analysis(submission)

    return results

if __name__ == '__main__' : 
    submission = get_submission("Random")
    results=run_analysis(submission)