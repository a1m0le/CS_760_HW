import numpy as np
from numpy import random
import math

import matplotlib.pyplot as plt




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
        empty_dict[point] = 0
    return truth_dict, empty_dict



def pick_center(samples):
    choices = random.choice(len(samples), 3, replace=False)
    c1 = samples[choices[0]]
    c2 = samples[choices[1]]
    c3 = samples[choices[2]]
    #print(c1)
    #print(c2)
    #print(c3)
    return [np.array(c1), np.array(c2), np.array(c3)]


def reassignment(toprocess, centers):
    memlists = [[],[],[]]
    for pt in toprocess:
        x_coord = np.array(pt)
        new_c = None
        curr_dist = None
        for i in range(0, len(centers)):
            c_coord = np.array(centers[i])
            dist = np.linalg.norm(x_coord - c_coord)
            if new_c is None:
                new_c = i+1
                curr_dist = dist
            elif dist < curr_dist:
                new_c = i+1
                curr_dist = dist
        toprocess[pt] = new_c
        memlists[new_c-1].append(x_coord)
    return memlists


def trueassignment(truth, centers):
    d ={"a":0, "b":1, "c":2}
    memlists = [[],[],[]]
    for pt in truth:
        x_coord = np.array(pt)
        c = truth[pt]
        i = d[c]
        memlists[i].append(x_coord)
    return memlists



def update_centers(memlists):
    new_c1 = np.mean(np.array(memlists[0]), axis=0)
    new_c2 = np.mean(np.array(memlists[1]), axis=0)
    new_c3 = np.mean(np.array(memlists[2]), axis=0)
    return (new_c1, new_c2, new_c3)




def visualize(memlists):
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
    plt.show()


def get_objective(toprocess, centers):
    total = 0
    for pt in toprocess:
        x_coord = np.array(pt)
        its_c = toprocess[pt]
        c_coord = centers[its_c-1]
        dist = np.linalg.norm(x_coord - c_coord)
        val = dist * dist # NOTE: objective has its squared.
        total += val
    return total



def get_accuracy(toprocess, truths, centers):
    truecenters = [np.array([-1,-1]),np.array([1,-1]), np.array([0,1])]
    truelabels = ["a","b","c"]
    mapping = []
    for i in range(0, len(centers)):
        res = []
        for j in range(0, len(truecenters)):
            dist = np.linalg.norm(centers[i] - truecenters[j])
            res.append(dist)
        mapping.append(np.argmin(res))
    total = 0
    correct = 0
    for pt in toprocess:
        predicted = toprocess[pt]-1
        mapped = truelabels[mapping[predicted]]
        if truths[pt] == mapped:
            correct += 1
        total += 1
    return correct/total*100




def run(sigma):
    truths, toprocess = load_data(sigma)
    # 1, 2, 3 for index 0, 1, 2
    centers = pick_center(list(toprocess.keys()))
    while True:
        memlists = reassignment(toprocess, centers)
        new_centers = update_centers(memlists)
        #print(new_centers)
        if np.array_equal(new_centers, centers):
            break
        else:
            centers = new_centers
    #print(centers)
    #tml = trueassignment(truths, centers)
    visualize(memlists)
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
