import numpy as np
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
from numpy import random
import math




Xs = []
Ys = []

with open("LI_train.txt", "r") as input_file:
        all_lines = input_file.readlines()
        for line in all_lines:
            comps = line.split()
            x = float(comps[0])
            y = float(comps[1])
            Xs.append(x)
            Ys.append(y)



nXs = []
add_noise = True
var = 10

for x in Xs:
    if add_noise:
        new_x = x + random.normal(0,math.sqrt(var))
        nXs.append(new_x)
    else:
        nXs.append(x)



desired_count = len(nXs)

li_poly = lagrange(nXs[:desired_count], Ys[:desired_count])


li_coef = li_poly.coef[::-1]

desired_order = len(li_coef)

li_coef = li_coef[:desired_order]
f = Polynomial(li_coef)
#print(f)


x_range = np.arange(0, 24, 1)
plt.ylim(-2.5,2.5)
plt.scatter(Xs, Ys)
plt.scatter(nXs, Ys)
plt.plot(x_range, f(x_range))
plt.show()


# compute the trainn erros
train_diffs = []
for i in range(0, len(Xs)):
    y = Ys[i]
    fx = f(Xs[i])
    diff = fx - y 
    train_diffs.append(diff)

train_diffs = np.array(train_diffs)
train_mse_error = np.square(train_diffs).mean()
print("train_error: "+str(train_mse_error))


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
print(" test_error: "+str(test_mse_error))
