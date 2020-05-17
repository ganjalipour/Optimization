import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
import random as random
import math

# problem definition
costfunction = lambda x: sum(x)

def roulettewheelSelection(p):
    r = random.random()
    cumsum = np.cumsum(p)
    y = (cumsum<r)
    x= [i for i in y if i==True]
    return len(x)-1
   
def singlepointcrossover(x1,x2):
    nvar = len(x1)
    c = np.random.randint(1,nvar-2)
    y1 = x1[:c] + x2[c:]
    y2 = x2[:c] + x1[c:]
    return (y1,y2)

def mutate(x):
    nVar = len(x)
    j = np.random.randint(1,nVar-2)
    y = x
    y[j] = 1- x[j]
    return y


nVar = 20               # number of decision varaibles
varSize = (1,nVar)
# GA parameters
maxIt= 1000              # max iteration number
nPop = 20                # number population

Pc = 0.8                # crossover percentage
Nc = 2*round(Pc*nPop/2) # number of offsprings = number of parents

Pm = 0.3                # mutation percentage
Nm = round(Pm*nPop)     # number of mutants

beta = 1 # selection pressure

# initialization
# class Individual:
#     position = []
#     cost = []
# pop = [Individual() for i in range(nPop)]

Pop_position = [[] for i in range(nPop)]
Pop_cost = [[] for i in range(nPop)]

for i in range(nPop):
    Pop_position[i] = np.random.randint(0,2,nVar).tolist()
    Pop_cost[i] = costfunction(Pop_position[i])

# sort population
sorted_Pop_cost = np.sort(Pop_cost)
sorted_index = np.argsort(Pop_cost)

# store best solution
bestSol = sorted_Pop_cost[0]
bestCostMemory = []


# main loop
for it in range(maxIt):

    # crossover 2 parent => 2 child
    PopCross1_position = [[] for i in range(int(Nc/2))]
    PopCross1_cost = [[] for i in range(int(Nc/2))]

    PopCross2_position = [[] for i in range(int(Nc/2))]
    PopCross2_cost = [[] for i in range(int(Nc/2))]

    for k in range(int(Nc/2)):
            
        # calculate selection probablity
        # p = [math.exp(-beta*i) for i in Pop_cost]
        # sump = sum(p)
        # p = [elem/sump for elem in p]
        # i1 = roulettewheelSelection(p)

        # select first parent
        i1 = np.random.randint(0,nPop)
        p1 = Pop_position[i1]

        # select second parent
        i2 = np.random.randint(0,nPop)
        p2 = Pop_position[i2]

        (PopCross1_position[k],PopCross2_position[k]) = singlepointcrossover(p1,p2)
        PopCross1_cost[k] = costfunction(PopCross1_position[k])
        PopCross2_cost[k] = costfunction(PopCross2_position[k])

    popC_postion = PopCross1_position + PopCross2_position 
    popC_cost =  PopCross1_cost + PopCross2_cost

    # mutation
    popM_position = [[] for i in range(Nm)]
    popM_cost = [[] for i in range(Nm)]
    for k in range(Nm):
        i = np.random.randint(0,Nm)
        p = Pop_position[i]
        popM_position[k] = mutate(p)
        popM_cost[k] = costfunction(popM_position[k])

    # merge
    Pop_position = Pop_position + popC_postion + popM_position
    Pop_cost = Pop_cost + popC_cost + popM_cost

    # sort population
    sorted_Pop_cost = np.sort(Pop_cost)
    sorted_index = np.argsort(Pop_cost)

    Pop_position = [Pop_position[i] for i in sorted_index[0:nPop]]
    Pop_cost = sorted_Pop_cost[0:nPop].tolist()

    bestCostMemory =bestCostMemory+ [Pop_cost[0]]


plt.plot(bestCostMemory) 
plt.show()


