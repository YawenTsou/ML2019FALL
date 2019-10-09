#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
from sklearn.neural_network import MLPRegressor
import sys

def testing_preprocess(group):
    group = group.drop(['id','測項'], axis = 1)
    for i in group.columns:
        group[i] = group[i].apply(lambda x: x.rstrip('#*xA'))
        group[i] = group[i].apply(lambda x: x.replace('NR', '0'))
        group[i] = group[i].astype(float)
        
    tmp = [list(x) for x in list(group.values)]
    return [j for sub in tmp for j in sub]

def test_data(test):
    test = test.fillna('0')
    group = test.groupby('id', as_index = False)
    
    return [key for key, y in group], [testing_preprocess(y) for key, y in group] 

if __name__ == "__main__":
    test = pd.read_csv(sys.argv[1])
    with open('MLP(100,50,50)_2.pickle', 'rb') as f:
        model = pickle.load(f)
        
    key, test_x = test_data(test)
    predicted = model.predict(test_x)

    end = pd.DataFrame({'id':key, 'value': predicted})
    end.to_csv(sys.argv[2], index = False)
