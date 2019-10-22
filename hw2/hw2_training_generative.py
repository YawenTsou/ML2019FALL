import numpy as np
import math
import pandas as pd
import pickle
import sys

def sigmoid(z):
    res =  1.0 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-6, 1-(1e-6))

def train(x_train, y_train):
    cnt1 = 0
    cnt2 = 0
    
    mu1 = np.zeros((dim,))
    mu2 = np.zeros((dim,))
    
    for i in range(x_train.shape[0]):
        if y_train[i] == 1:
            cnt1 += 1
            mu1 += x_train[i]
        else:
            cnt2 += 1
            mu2 += x_train[i]
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((dim,dim))
    sigma2 = np.zeros((dim,dim))
    for i in range(x_train.shape[0]):
        if y_train[i] == 1:
            sigma1 += np.dot(np.transpose([x_train[i] - mu1]), [(x_train[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([x_train[i] - mu2]), [(x_train[i] - mu2)])
    sigma1 /= cnt1
    sigma2 /= cnt2

    
    share_sigma = (cnt1 / x_train.shape[0]) * sigma1 + (cnt2 / x_train.shape[0]) * sigma2
    return mu1, mu2, share_sigma, cnt1, cnt2

def predict(x_test, mu1, mu2, share_sigma, N1, N2):
    sigma_inverse = np.linalg.inv(share_sigma)

    w = np.dot( (mu1-mu2), sigma_inverse)
    b = (-0.5) * np.dot(np.dot(mu1.T, sigma_inverse), mu1) + (0.5) * np.dot(np.dot(mu2.T, sigma_inverse), mu2) + np.log(float(N1)/N2)

    z = np.dot(w, x_test.T) + b
    pred = sigmoid(z)
    return pred

if __name__ == '__main__':
    train_x = pd.read_csv(sys.argv[1])
    train_y = pd.read_csv(sys.argv[2], header = None)

    train_x = train_x.values
    train_y = train_y.values
    train_y = train_y.reshape(-1)
    
    mu1, mu2, shared_sigma, N1, N2 = train(train_x, train_y)

    y = predict(train_x, mu1, mu2, shared_sigma, N1, N2)
    y = [1 if x >= 0.5 else 0 for x in y]
    result = (train_y == y)

    print('Train acc = %f' % (float(result.sum()) / result.shape[0]))
    
    par = [mu1, mu2, shared_sigma, N1, N2]
    with open('generative_reproduce.pkl', 'wb') as f:
        pickle.dump(par, f)