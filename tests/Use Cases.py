# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:00:52 2022

@author: mgrace31
"""

'Testing different HuggigFace models on the datasets'

from src.data.cold import COLD, COLDSubmissionObject
from src.analysis.cold import COLDAnalysis
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
    cold._load_data()
    dataset = cold.data()
    dataset.drop(dataset.tail(1).index,inplace=True)
    
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
    submission = cold.submit(dataset, preds, map=MODELS[model_name]["map"])
    return submission

