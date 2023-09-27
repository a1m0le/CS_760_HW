import os
import numpy as np

noise_vals = np.arange(8, 16, 1)
for n in noise_vals:
    if n > 0:
        command = "python3 LI.py "+str(n)
    else:
        command = "python3 LI.py"
    os.system(command)
