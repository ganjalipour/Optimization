import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
import random as random
import math

A =  np.array([[3,-1,-2],[-1,1,3],[-2,3,5]])
eigens = np.linalg.eigvals(A)
eigenvectors = np.linalg.eigh(A)
print(eigenvectors)

from scipy.sparse import spdiags
n = 1000
a_3= np.full((1,n),0.8)
a_2=np.full((1,n),0.42)
a_1=np.full((1,n),1.2)

a0=np.arange(start=3, stop=n+3, step=1).reshape(a_1.shape)
#a0=np.full((1,n),0.2)
a1=np.full((1,n),1.2)
a2=np.full((1,n),0.42)
a3=np.full((1,n),0.8)

data = np.array([[a_3],[a_2],[a_1],[a0], [a1],[a2],[a3]]).reshape((7,n))
diags = np.array([-3,-2,-1,0, 1,2 , 3])
datatestOstad = spdiags(data, diags, n, n).toarray()
print(datatestOstad)