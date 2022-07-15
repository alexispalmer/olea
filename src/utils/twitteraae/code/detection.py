# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 09:13:27 2022

@author: mcg19
"""
import pandas as pd
from src.utils.twitteraae.code import predict
from src.utils import preprocess_text 

def get_aave_values(submission):
    predict.load_model()
    pt = preprocess_text.PreprocessText()
    processed_text = pt.execute(submission.submission[submission.text_column])
    aae = []
    for i in range(len(processed_text)):
        preds = predict.predict((processed_text[i].split()))
        if preds is None:
            preds = [0,0,0,0]  
        aae.append(preds[0])
    return aae
    
    
