# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:55:01 2022

@author: mcg19

TESTING FOR ANALYSIS
"""
from analysis.analysis import Analysis
import pandas as pd
import numpy as np
from metrics.metrics import Metrics


pd.set_option('display.max_columns', None)

#create cold
cold =pd.read_csv('data/cold-2016-all-annos.tsv',sep='\t',encoding='utf-8')

#create predicted labels
models = pd.read_csv('data/cold-2016-majVote-withResults.tsv',sep = '\t',encoding = 'utf-8')
models = models.drop(["DataSet", "Off","Slur","Nom", "Cat", "Text"],1)

extras = models[~models['ID'].isin(cold["ID"])]

cold = cold.merge(models,how = "inner")
cold = cold.rename({"Mod3":"pred"},axis = 1)

cold["pred"] = cold["pred"].replace(to_replace=['HOF', 'NOT'], value=["Y", "N"])

#create majority votes
OffMaj = []
SlurMaj =[]
NomMaj = []
DistMaj = []

for i in range((cold.shape[0])):
    offs = [cold.Off1[i],cold.Off2[i],cold.Off3[i]]
    slurs = [cold.Slur1[i],cold.Slur2[i],cold.Slur3[i]]
    noms = [cold.Nom1[i],cold.Nom2[i],cold.Nom3[i]]
    dists = [cold.Dist1[i],cold.Dist2[i],cold.Dist3[i]]
    
    OffMaj.append(max(set(offs), key=offs.count))
    SlurMaj.append(max(set(slurs), key=slurs.count))
    NomMaj.append(max(set(noms), key=noms.count))
    DistMaj.append(max(set(dists), key=dists.count))

#add to dataset    
cold["OffMaj"] = OffMaj
cold["SlurMaj"] = SlurMaj
cold["NomMaj"] = NomMaj
cold["DistMaj"] = DistMaj


#create class
show_examples = 1
num_annotators = 1

myAnalysis= Analysis(cold,show_examples)

#run analysis
str_len_results = myAnalysis.check_string_len()
hashtag_results = myAnalysis.check_substring("#")
quotes_results = myAnalysis.check_substring('"')
anno_agree_results = myAnalysis.check_anno_agreement(num_annotators)
cat_results = myAnalysis.category_performance()
anno_fg_results = myAnalysis.anno_fine_grained()
