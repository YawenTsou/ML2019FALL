import numpy as np
import math
import pandas as pd
import pickle
import sys

def predict(x_test, mu1, mu2, share_sigma, N1, N2):
    sigma_inverse = np.linalg.inv(share_sigma)

    w = np.dot( (mu1-mu2), sigma_inverse)
    b = (-0.5) * np.dot(np.dot(mu1.T, sigma_inverse), mu1) + (0.5) * np.dot(np.dot(mu2.T, sigma_inverse), mu2) + np.log(float(N1)/N2)

    z = np.dot(w, x_test.T) + b
    pred = sigmoid(z)
    return pred

if __name__ == '__main__':
    test_x = pd.read_csv(sys.argv[1])
    test_x = test_x.values
    
    with open('generative.pkl', 'rb') as f:
        par = pickle.load(f)
        
    y = predict(test_x, par[0], par[1], par[2], par[3], par[4])
    y = [1 if x >= 0.5 else 0 for x in y]
    
    end = pd.DataFrame({'id': range(1,len(test_x)+1), 'label': y})
    end.to_csv(sys.argv[2], index = False)