#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import sys

def preprocess(train):
    train['month'] = train['日期'].apply(lambda x:x.split('/')[1])
    train = train.drop(['日期', '測項'], axis = 1)
    train = train.fillna('0')
    for i in train.columns:
        train[i] = train[i].apply(lambda x: x.rstrip('#*xA'))
        train[i] = train[i].apply(lambda x: x.replace('NR', '0'))
        train[i] = train[i].astype(float)
        
    return train

def reshape(group):
    group = group.drop('month', axis = 1)
    new = [list(x) for x in list(group.iloc[:18].values)]
    flag = 18
    while flag < len(group):
        tmp = [list(x) for x in list(group.iloc[flag:flag+18].values)]
        for i in range(len(new)):
            new[i].extend(tmp[i])
        flag += 18
    return new    

def feature(group):
    result = []
    y = []
    flag = 0
    while flag <= len(group[0])-10:
        if group[9][flag+9] <= 2 or group[9][flag+9] > 100:
            flag += 1
            continue
        tmp = [x[flag:flag+9] for x in group]
        result.append([j for sub in tmp for j in sub])
        y.append(group[9][flag+9])
        flag += 1
    return result, y


if __name__ == "__main__":
    train = pd.read_csv(sys.argv[1])
    train1 = pd.read_csv(sys.argv[2])
    train = preprocess(train)
    train1 = preprocess(train1)
    group = train.groupby('month', as_index = False)
    group1 = train1.groupby('month', as_index = False)
    data = [reshape(y) for month, y in group]
    data1 = [reshape(y) for month, y in group1]
    data = data + data1
    result = [feature(x) for x in data]
    train_x = [j for sub in result for j in sub[0]]
    train_y = [j for sub in result for j in sub[1]]
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    MLP = MLPRegressor(hidden_layer_sizes=(100,50,50))
    MLP.fit(train_x, train_y)
#     mse = np.mean((train_y - MLP.predict(train_x))**2)
#     print(mse)

    with open('MLP(100,50,50)_reproduce.pickle', 'wb') as f:
        pickle.dump(MLP, f)

