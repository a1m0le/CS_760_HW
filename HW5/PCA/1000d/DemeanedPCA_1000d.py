import sys
import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt



FILENAME = "data1000D.csv" 

REDUCE_TO = 30



def load_data(filename):
    with open(filename) as f:
        lines = f.readlines()
    X = []
    for l in lines:
        nstr = l.split(",")
        ns = []
        for n in nstr:
            ns.append(float(n))
        X.append(ns)
    X = np.array(X)
    return X




def run(X, d):
    mean = np.mean(X, axis=0)
    newX = []
    for i in range(0, X.shape[0]):
        newX.append(X[i]-mean)
    newX = np.array(newX)
    _, lambs, As = la.svd(newX, full_matrices=False)
    A = As[:d]
    A = np.transpose(A)
    Z = newX @ A
    RawReX = Z @ np.transpose(A) 
    ReX = []
    for i in range(0, X.shape[0]):
        ReX.append(RawReX[i]+mean)
    ReX = np.array(ReX)
    return Z, A, lambs, ReX



def get_recon_error(X, ReX):
    error = 0
    for i in range(0, X.shape[0]):
        diff = X[i] - ReX[i]
        e = np.dot(diff, diff)
        error += e
    return error


if __name__=="__main__":
    X = load_data(FILENAME)
    Z, A, lambs, ReX = run(X, REDUCE_TO)
    error = get_recon_error(X, ReX)
    print("Reconstruction error = "+str(error))


