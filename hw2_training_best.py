import numpy as np
import math
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import sys

def normalize(train_x, test_x):
    x_all = np.concatenate((train_x, test_x))
    mean = np.mean(x_all, axis = 0)
    std = np.std(x_all, axis = 0)

    index = [0, 1, 3, 4, 5]
    mean_vec = np.zeros(x_all.shape[1])
    std_vec = np.ones(x_all.shape[1])
    mean_vec[index] = mean[index]
    std_vec[index] = std[index]
    
    x_all_nor = (x_all - mean_vec) / std_vec

    x_train_nor = x_all_nor[0:train_x.shape[0]]
    x_test_nor = x_all_nor[train_x.shape[0]:]
    return x_train_nor, x_test_nor

if __name__ == '__main__':
    train_x = pd.read_csv(sys.argv[1])
    test_x = pd.read_csv(sys.argv[2])
    train_y = pd.read_csv(sys.argv[3], header = None)

    train_x = train_x.values
    test_x = test_x.values
    train_y = train_y.values
    train_y = train_y.reshape(-1)
    
    train_x, test_x = normalize(train_x, test_x)
    
    rf = RandomForestClassifier(n_estimators = 200)
    rf.fit(train_x, train_y)
    wrong = len([x for x in (train_y - rf.predict(train_x)) if x != 0])
    print(1-(wrong/len(train_x)))
    
    with open('rf(200)_reproduce.pickle', 'wb') as f:
        pickle.dump(rf, f)