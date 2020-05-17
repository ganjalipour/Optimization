import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
import random as random
import math


# problem definition
def MOP2(x):
    x = np.array(x)
    n= len(x)
    z1 = 1 - math.exp(-sum((x-1/math.sqrt(n))**2))
    z2 = 1 - math.exp(-sum((x+1/math.sqrt(n))**2))
    return [z1,z2]

costfunction = lambda x: MOP2(x)

