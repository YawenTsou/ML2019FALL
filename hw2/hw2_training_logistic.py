import numpy as np
import pandas as pd
import sys

def sigmoid(z):
    res =  1.0 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))

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

def train(x, y):
    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    x = x[index]
    y = y[index]
    
    batch_size = 32
    w = np.random.rand(x.shape[1])
    b = 0
    lr = 0.3
    iteration = 1000

    lr_b = 0
    lr_w = np.ones(x.shape[1])
    
    print('Start training...')
    for i in range(iteration):
        z = np.dot(x, w) + b
        pre = sigmoid(z)
        loss = y - pre

        w_grad = np.sum(-1 * loss.reshape(len(loss), 1) * x, axis = 0)
        b_grad = np.sum(-1 * loss)


        lr_b = lr_b + b_grad ** 2
        lr_w = lr_w + w_grad ** 2

        b = b - (lr/np.sqrt(lr_b)) * b_grad
        w = w - (lr/np.sqrt(lr_w)) * w_grad
    
    print('Finish！')
    return w, b


if __name__ == '__main__':
    train_x = pd.read_csv(sys.argv[1])
    test_x = pd.read_csv(sys.argv[2])
    train_y = pd.read_csv(sys.argv[3], header = None)

    train_x = train_x.values
    test_x = test_x.values
    train_y = train_y.values
    train_y = train_y.reshape(-1)
    
    train_x, test_x = normalize(train_x, test_x)
    
    w, b = train(train_x, train_y)
    predicate = np.dot(train_x, np.transpose(w)) + b
    predicate = sigmoid(predicate)
    predicate = [1 if x >= 0.5 else 0 for x in predicate]
    wrong = len([x for x in (train_y-predicate) if x != 0])
    print(1-(wrong/len(train_y)))
    
    np.save('weight_base_reproduce', np.append(w, b))