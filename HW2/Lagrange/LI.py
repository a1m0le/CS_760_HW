import numpy as np
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
from numpy import random
import math
import sys

Xs = []
Ys = []


plt.figure(figsize=(8, 5))

with open("LI_train.txt", "r") as input_file:
        all_lines = input_file.readlines()
        for line in all_lines:
            comps = line.split()
            x = float(comps[0])
            y = float(comps[1])
            Xs.append(x)
            Ys.append(y)

nXs = []

add_noise = False
std = 0
if len(sys.argv) > 1:
    add_noise = True
    std = float(sys.argv[1])

for x in Xs:
    if add_noise:
        new_x = x + random.normal(0,std)
        nXs.append(new_x)
    else:
        nXs.append(x)

li_poly = lagrange(nXs, Ys)
li_coef = li_poly.coef[::-1]
f = Polynomial(li_coef)
#print(f)


x_range = np.arange(min(nXs), max(nXs), 0.01)
plt.xlim(min(nXs),max(nXs))
plt.ylim(-1.5, 1.5)

plt.scatter(Xs, Ys)
plt.scatter(nXs, Ys)
plt.plot(x_range, f(x_range), linewidth=0.5, color="green")


plt.title("std = "+str(std))
plt.savefig("std_"+str(std)+".png")


# compute the traing erros
train_diffs = []
for i in range(0, len(Xs)):
    y = Ys[i]
    fx = f(Xs[i])
    diff = fx - y 
    train_diffs.append(diff)

train_diffs = np.array(train_diffs)
train_mse_error = np.square(train_diffs).mean()

tXs = []
tYs = []

with open("LI_test.txt", "r") as input_file:
        all_lines = input_file.readlines()
        for line in all_lines:
            comps = line.split()
            x = float(comps[0])
            y = float(comps[1])
            tXs.append(x)
            tYs.append(y)

# compute the test erros
test_diffs = []
for i in range(0, len(tXs)):
    y = tYs[i]
    fx = f(tXs[i])
    diff = fx - y 
    test_diffs.append(diff)

test_diffs = np.array(test_diffs)
test_mse_error = np.square(test_diffs).mean()
print(str(std)+","+str(train_mse_error)+","+str(test_mse_error))
