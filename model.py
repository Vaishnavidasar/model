#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import numpy as np
import pickle
os.getcwd()

with open('model_pickle','rb')as f:
    rf=pickle.load(f)

def func2(data):
    
    data2=data
    
    sent = data2.Sentence
    
    data2 = data2.loc[data2.nwords > 2, :]
    data2 = data2.loc[data2.nnumbers <= 3, :]
    data2 = data2.loc[data2.nwords <= 10, :]
    data2 = data2.drop(["Sentence"],axis=1)
    
    X_test = data2
    
    
    pred = rf.predict(X_test)

    data2['label'] = np.array(pred)
    data2.insert(loc=0, column='Sentence', value=sent)
    data2.to_csv('label.csv',index = False)
    return(data2)