import numpy as np
from numpy import random
import math


from scipy.stats import multivariate_normal as twoDgauss
import matplotlib.pyplot as plt

K = 3


class Z:

    def __init__(self, phi, mean, cov):
        self.phi = phi
        self.mean = mean
        self.cov = cov


def load_data(sigma):
    filename = "data/sigma_"+str(sigma)+".txt"
    truth_dict = {}
    empty_dict = {}
    with open(filename, "r") as f:
        lines = f.readlines()
    for l in lines:
        vals = l.split()
        point = (float(vals[0]),float(vals[1]))
        truth = vals[2]
        truth_dict[point] = truth
        empty_dict[point] = np.random.randint(3)
    return truth_dict, empty_dict



def pick_center(toprocess):
    centerZs = []
    for i in range(0, K):
        members = []
        for pt in toprocess:
            if toprocess[pt] == i:
                members.append(np.array(pt))
        phi = len(members) / len(toprocess)
        mean = np.mean(np.array(members), axis=0)
        cov = np.cov(np.array(members), rowvar=False)
        #print(phi)
        #print(mean)
        #print(cov)
        newcZ = Z(phi, mean, cov)
        centerZs.append(newcZ)
    return centerZs




def Estep(toprocess, centerZs):
    wijs = {}
    #print("#########################")
    for pt in toprocess:
        x_coord = np.array(pt)
        wsum = 0
        wjs = []
        for i in range(0, len(centerZs)):
            cZ = centerZs[i]
            #print(cZ.cov)
            likelihood = twoDgauss.pdf(x_coord, mean=cZ.mean, cov=cZ.cov)
            prior = cZ.phi
            wj = likelihood*prior
            wjs.append(wj)
            wsum += wj
        wjs = np.array(wjs) / wsum
        wijs[pt] = wjs
    return wijs



def trueassignment(truth, centers):
    d ={"a":0, "b":1, "c":2}
    memlists = [[],[],[]]
    for pt in truth:
        x_coord = np.array(pt)
        c = truth[pt]
        i = d[c]
        memlists[i].append(x_coord)
    return memlists



def Mstep(toprocess, centerZs, wijs):
    new_centerZs = []
    for i in range(0, K):
        z_toupdate = centerZs[i]
        wsum = 0
        pt_sum = 0 * z_toupdate.mean
        for pt in wijs:
            wjs = wijs[pt]
            wj = wjs[i]
            wsum += wj
            x_coord = np.array(pt)
            pt_sum = pt_sum + (wj * x_coord)
        new_phi = wsum / len(wijs)
        new_mean = pt_sum / wsum
        cov_sum = 0 * z_toupdate.cov
        for pt in wijs:
            wjs = wijs[pt]
            wj = wjs[i]
            x_coord = np.array(pt)
            cov_sum = cov_sum + (wj * (np.outer(x_coord-new_mean, x_coord-new_mean)))
        #print("-----------"+str(cov_sum.ndim)) 
        new_cov = cov_sum / wsum
        #print("-----------"+str(new_cov.ndim))
        new_Z = Z(new_phi, new_mean, new_cov)
        new_centerZs.append(new_Z)
    return new_centerZs



def classify(toprocess, centers):
    memlists = [[],[],[]]
    for pt in toprocess:
        x_coord = np.array(pt)
        new_c = None
        curr_like = None
        for i in range(0, len(centers)):
            cZ = centers[i]
            like = twoDgauss.pdf(x_coord, mean=cZ.mean, cov=cZ.cov)
            if new_c is None:
                new_c = i
                curr_like = like
            elif like > curr_like:
                new_c = i
                curr_like = like
        toprocess[pt] = new_c
        memlists[new_c].append(x_coord)
    return memlists


def visualize(memlists, filename):
    # firstly make a scatter plot
    fig, axes = plt.subplots()
    fig.set_figheight(9)
    fig.set_figwidth(16)
    
    colors = ["red","green","blue"]
    # process points
    for i in range(0, len(memlists)):
        x_1_y1 = []
        x_2_y1 = []
        for inst in memlists[i]:
            x_1_y1.append(inst[0])
            x_2_y1.append(inst[1])
        axes.scatter(x_1_y1, x_2_y1, color = colors[i])
    plt.savefig("gmm_visual/"+filename+".png")


def get_objective(toprocess, centers):
    total = 0
    for pt in toprocess:
        x_coord = np.array(pt)
        unlogged = 0
        for i in range(0, K):
            cZ = centers[i]
            likelihood = twoDgauss.pdf(x_coord, mean=cZ.mean, cov=cZ.cov)
            prior = cZ.phi
            unlogged += likelihood * prior
        logged = math.log(unlogged)
        total += logged
    return -1*total



def get_accuracy(toprocess, truths, centers):
    truecenters = [np.array([-1,-1]),np.array([1,-1]), np.array([0,1])]
    truelabels = ["a","b","c"]
    mapping = []
    for i in range(0, len(centers)):
        res = []
        for j in range(0, len(truecenters)):
            dist = np.linalg.norm(centers[i].mean - truecenters[j])
            res.append(dist)
        mapping.append(np.argmin(res))
    total = 0
    correct = 0
    for pt in toprocess:
        predicted = toprocess[pt]
        mapped = truelabels[mapping[predicted]]
        if truths[pt] == mapped:
            correct += 1
        total += 1
    return correct/total*100











def run(sigma):
    truths, toprocess = load_data(sigma)
    # 1, 2, 3 for index 0, 1, 2
    centers = pick_center(toprocess)
    counter = 0
    while True:
        wijs = Estep(toprocess, centers)
        new_centers = Mstep(toprocess, centers, wijs)
        counter = counter + 1
        #print(new_centers[0].mean,end="  ")
        #print(counter)
        centers = new_centers
        if counter % 40 == 0:
            print(new_centers[0].mean,end="  ")
            print(counter)
        if counter >= 800: #print and observer, at 800 iteration, the printable mean value no longer changes
            break 
    #print(centers)
    #tml = trueassignment(truths, centers)
    memlists = classify(toprocess, centers)
    visualize(memlists, "sigma_"+str(sigma)+"_gmm")
    tml = trueassignment(truths, centers)
    visualize(tml, "sigma_"+str(sigma)+"_truth")
    objective = get_objective(toprocess, centers)
    accuracy = get_accuracy(toprocess, truths, centers)
    return  objective, accuracy




if __name__=="__main__":
    sigma = [0.5, 1, 2, 4, 8]
    os = []
    accus = []
    for s in sigma:
        o, a = run(s)
        os.append(o)
        accus.append(a)
    print("Objective for each sigma:")
    print(os)
    print()
    print("Accuracy for each sigma (in percentage. '%' symbol not shown here):")
    print(accus)
