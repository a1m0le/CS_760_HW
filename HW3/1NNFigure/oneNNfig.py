import matplotlib.pyplot as plt
import numpy as np
import math


class points:
    
    def __init__(self, x1, x2, y):
        self.x1 = x1
        self.x2 = x2
        self.y = y




training_data = []
lines = None
with open("D2z.txt","r") as datafile:
    lines = datafile.readlines()

for l in lines:
    strs = l.split()
    x1 = float(strs[0])
    x2 = float(strs[1])
    y = int(strs[2])
    new_pt = points(x1, x2, y)
    training_data.append(new_pt)


left = np.arange(-2,0,0.1)
right = np.arange(0,2.1,0.1)
test_range = np.concatenate((left,right))

test_points = []

for test1 in test_range:
    for test2 in test_range:
        #print(str(test1)+", "+str(test2))
        # use stupid method
        min_dist = 0
        closest_pt = None
        for pt in training_data:
            x1_diff = test1 - pt.x1
            x2_diff = test2 - pt.x2
            dist = math.sqrt(x1_diff * x1_diff + x2_diff * x2_diff)
            if closest_pt is None:
                min_dist = dist
                closest_pt = pt
            else:
                if dist < min_dist:
                    min_dist = dist
                    closest_pt = pt

        new_test_pt = points(test1, test2, closest_pt.y)
        test_points.append(new_test_pt)


# now the plotting

trainX1_1 = []
trainX2_1 = []
for pt in training_data:
    if pt.y == 1:
        trainX1_1.append(pt.x1)
        trainX2_1.append(pt.x2)

trainX1_0 = []
trainX2_0 = []
for pt in training_data:
    if pt.y == 0:
        trainX1_0.append(pt.x1)
        trainX2_0.append(pt.x2)

testX1_1 = []
testX2_1 = []
for pt in test_points:
    if pt.y == 1:
        testX1_1.append(pt.x1)
        testX2_1.append(pt.x2)

testX1_0 = []
testX2_0 = []
for pt in test_points:
    if pt.y == 0:
        testX1_0.append(pt.x1)
        testX2_0.append(pt.x2)


plt.figure(figsize=(6,6))

plt.xlim(-2,2)
plt.ylim(-2,2)


plt.scatter(trainX1_1, trainX2_1, c="black", linewidths=1, marker="*")
plt.scatter(trainX1_0, trainX2_0, linewidths=0.5, c="black",marker="x")
plt.scatter(testX1_1, testX2_1, c="green",  linewidths=0.5, marker="+")
plt.scatter(testX1_0, testX2_0, c="red", linewidths=0.5,  marker="+")

#plt.show()
plt.savefig("grid.png")






