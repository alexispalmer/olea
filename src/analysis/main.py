# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:55:01 2022

@author: mcg19

TESTING FOR ANALYSIS
"""
from analysis.analysis import Analysis
import pandas as pd
import numpy as np



#create cold
cold =pd.read_csv('data/cold_mock_data.tsv',sep='\t',encoding='utf-8')
#create predicted labels
y = np.random.choice(["Y","N"],size=(1,len(cold)))

#create majority votes
OffMaj = []
SlurMaj =[]
NomMaj = []

for i in range((cold.shape[0])):
    offs = [cold.Off1[i],cold.Off2[i],cold.Off3[i]]
    slurs = [cold.Slur1[i],cold.Slur2[i],cold.Slur3[i]]
    noms = [cold.Nom1[i],cold.Nom2[i],cold.Nom3[i]]
    
    OffMaj.append(max(set(offs), key=offs.count))
    SlurMaj.append(max(set(offs), key=slurs.count))
    NomMaj.append(max(set(offs), key=noms.count))

#add to dataset    
cold["OffMaj"] = OffMaj
cold["SlurMaj"] = SlurMaj
cold["NomMaj"] = NomMaj
cold["pred"] = y.T

#create class
show_examples = True
myAnalysis= Analysis(cold,show_examples)
num_annotators = 3

# #run analysis
str_len_results = myAnalysis.check_string_len()
print("\n")
hashtag_results = myAnalysis.check_substring("#")
print("\n")
quotes_results = myAnalysis.check_substring('"')
print("\n")
analysis_results = myAnalysis.check_anno_agreement(num_annotators)