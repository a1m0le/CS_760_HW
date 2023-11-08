import sys
import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt



FILENAME = "data1000D.csv" 

REDUCE_TO = 30

RATIO_THRESHOLD = 0.005

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



def obtain_Y(X):
    # firstly get b (or b^T)
    oneone = np.zeros(X.shape[0]) + 1
    nbT = oneone @ X
    bT = nbT / X.shape[0]
    onebT = np.outer(oneone,bT)
    Y = X - onebT
    return Y, bT



def run(X, d):
    Y, bT = obtain_Y(X) 
    U, lambs, As = la.svd(Y, full_matrices=False)
    A = As[:d]
    A = np.transpose(A)
    oneone = np.zeros(X.shape[0]) + 1
    onebT = np.outer(oneone,bT)
    Z = Y @ A
    ReX = Z @ np.transpose(A) + onebT
    return Z, U, As, bT, lambs, ReX


def get_recon_error(X, ReX):
    error = 0
    for i in range(0, X.shape[0]):
        diff = X[i] - ReX[i]
        e = np.dot(diff, diff)
        error += e
    return error


def visualize_lambs(lambs):
    plt.figure(figsize=(10,6))
    plt.title("Lambda values")
    plt.plot(lambs)
    plt.savefig("lamb_decision/lamb_vals.png")
    lsum = np.sum(lambs)
    ratios = lambs / lsum
    plt.figure(figsize=(10,6))
    plt.title("Lambda vals' ratio among all vals")
    plt.bar(np.arange(len(lambs)), ratios)
    plt.savefig("lamb_decision/lamb_ratios.png")
    cutoff = 0
    for i in range(0, len(ratios)):
        if ratios[i] < RATIO_THRESHOLD:
            cutoff = i
            break
    print("We should reduce it to "+str(cutoff)+"-D space")


if __name__=="__main__":
    X = load_data(FILENAME)
    Z, U, A, bT, lambs, ReX = run(X, REDUCE_TO)
    visualize_lambs(lambs)
    error = get_recon_error(X, ReX)
    print("Reconstruction error = "+str(error))


