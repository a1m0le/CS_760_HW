import numpy as np
from numpy import random
import math


sigma = [0.5, 1, 2, 4, 8]

pa_mean = np.array([-1,-1])
pa_cov = np.array([[2,0.5],[0.5,1]])

pb_mean = np.array([1,-1])
pb_cov = np.array([[1,-0.5],[-0.5,2]])

pc_mean = np.array([0,1])
pc_cov = np.array([[1,0],[0,2]])


for s in sigma:
    filename = "sigma_"+str(s)+".txt"
    outf = open(filename, "w+")
    for i in range(0,100):
        out_a = random.multivariate_normal(pa_mean, s * pa_cov)
        outf.write(str(out_a[0])+" "+str(out_a[1])+" a\n")
        out_b = random.multivariate_normal(pb_mean, s * pb_cov)
        outf.write(str(out_b[0])+" "+str(out_b[1])+" b\n")
        out_c = random.multivariate_normal(pc_mean, s * pc_cov)
        outf.write(str(out_c[0])+" "+str(out_c[1])+" c\n")
    outf.close()


