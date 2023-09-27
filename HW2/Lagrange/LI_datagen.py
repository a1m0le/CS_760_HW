import math
import random


a = 0
b = 20

with open("LI_train.txt","w+") as train_f:
    for i in range(0, 100):
        x = random.uniform(a, b)
        y = math.sin(x)
        train_f.write(str(x)+" "+str(y)+"\n")


with open("LI_test.txt","w+") as test_f:
    for i in range(0, 25):
        x = random.uniform(a, b)
        y = math.sin(x)
        test_f.write(str(x)+" "+str(y)+"\n")





