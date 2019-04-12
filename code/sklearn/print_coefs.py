# encoding: utf-8 

import numpy as np

import sklearn as sk

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

from sklearn.externals import joblib    
    
def get_mlp(filename):
    pipe = joblib.load(filename)
    mlp = pipe.steps[0][1]
    
    return mlp
    

def print_w(mlp, basename):
    coefs = mlp.coefs_
    for i in range(len(coefs)):
        w = coefs[i]
        filename = "%s%s.txt" % (basename, i+1)
        np.savetxt(filename, w, delimiter=',', newline='],\n[', header='[\n[', footer=']', comments='')
        
def print_b(mlp, basename):
    coefs = mlp.intercepts_
    for i in range(len(coefs)):
        w = coefs[i]
        filename = "%s%s.txt" % (basename, i+1)
        np.savetxt(filename, w, delimiter=',', newline=',\n', header='[\n', footer=']', comments='')


def print_random_results(mlp, diff=0.9):
        
    tmp = 2*np.random.uniform(size=20) - 1
    res = (mlp.predict_proba(tmp.reshape(-1, 20)))
    x1, x2 = res[0][0], res[0][1]
    
    if np.abs(x1 - x2) < diff:
        print(repr(tmp))
        print(x1, x2)

if __name__ == "__main__":
    mlp = get_mlp('model.pkl')
    print_w(mlp, 'w_')
    print_b(mlp, 'b_')
    
    for i in range(10):
        print_random_results(mlp)
        # print
