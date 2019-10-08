#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
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
    w = np.load('weight_base_mini.npy')
    b = w[-1]
    w = w[:-1]
    key, test_x = test_data(test)

    test_y = np.dot(test_x, w) + b

    end = pd.DataFrame({'id': key, 'value': test_y})
    end.to_csv(sys.argv[2], index = False)
