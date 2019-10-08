#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
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


def minibatch(x, y):
    # 打亂data順序
    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    x = x[index]
    y = y[index]
    
    # 訓練參數以及初始化
    batch_size = 64
    lr = 1e-3
    lam = 0.001
    beta_1 = np.full(x[0].shape, 0.9).reshape(-1, 1)
    beta_2 = np.full(x[0].shape, 0.99).reshape(-1, 1)
    w = np.full(x[0].shape, 0.1).reshape(-1, 1)
    bias = 0.1
    m_t = np.full(x[0].shape, 0).reshape(-1, 1)
    v_t = np.full(x[0].shape, 0).reshape(-1, 1)
    m_t_b = 0.0
    v_t_b = 0.0
    t = 0
    epsilon = 1e-8
    
    for num in range(1000):
        for b in range(int(x.shape[0]/batch_size)):
            t+=1
            x_batch = x[b*batch_size:(b+1)*batch_size]
            y_batch = y[b*batch_size:(b+1)*batch_size].reshape(-1,1)
            loss = y_batch - np.dot(x_batch,w) - bias
            
            # 計算gradient
            g_t = np.dot(x_batch.transpose(),loss) * (-2) +  2 * lam * np.sum(w)
            g_t_b = loss.sum(axis=0) * (2)
            m_t = beta_1*m_t + (1-beta_1)*g_t 
            v_t = beta_2*v_t + (1-beta_2)*np.multiply(g_t, g_t)
            m_cap = m_t/(1-(beta_1**t))
            v_cap = v_t/(1-(beta_2**t))
            m_t_b = 0.9*m_t_b + (1-0.9)*g_t_b
            v_t_b = 0.99*v_t_b + (1-0.99)*(g_t_b*g_t_b) 
            m_cap_b = m_t_b/(1-(0.9**t))
            v_cap_b = v_t_b/(1-(0.99**t))
            w_0 = np.copy(w)
            
            # 更新weight, bias
            w -= ((lr*m_cap)/(np.sqrt(v_cap)+epsilon)).reshape(-1, 1)
            bias -= (lr*m_cap_b)/(math.sqrt(v_cap_b)+epsilon)
            

    return w, bias

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

    w, b = minibatch(train_x, train_y)
#     mse = np.mean((train_y - (np.dot(train_x, w) + b))**2)
#     print(mse)

    np.save('weight_base_mini_reproduce', np.append(w, b))

