import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
import random as random

# problem definition changed
def sphere(x):
    x=np.array(x)
    return np.sum(x**2)

costfunction = sphere

nVar = 5 # number of decision vars
varMin = -10
varMax = 10
maxIt = 1000
nPop = 20     #population size
w = 1 # inertia wieght
c1 = 2 # personal learning coefficient
c2 = 2 # global learning coefficient
wdamping = 0.99

################ constriction coefficients
phi1 = 2.05
phi2 = 2.05
phi = phi1+phi2
chi = 2/(phi - 2 + np.sqrt(phi**2 - 4*phi))
w = chi # inertia wieght
c1 = chi*phi1 # personal learning coefficient
c2 = chi*phi2 # global learning coefficient
wdamping = 1
#################


# initialization
class Particle:
    position = []
    cost = []
    velocity = []
    best_position = []
    best_cost = []

#Particles = np.matlib.repmat(Particle,nPop,1)
Particles = [Particle() for p in range(nPop)]

globalBest_cost = float("inf")
globalBest_position =[] 

BestCostMemory = []
BestPosionMemory = [np.empty(nVar)]

for i in range(nPop):
    Particles[i].position = np.random.uniform(varMin,varMax,nVar)
    Particles[i].velocity = np.zeros(nVar)
    Particles[i].cost = costfunction(Particles[i].position)
    Particles[i].best_position = Particles[i].position
    Particles[i].best_cost = Particles[i].cost

    if Particles[i].best_cost < globalBest_cost:
        globalBest_cost = Particles[i].best_cost
        globalBest_position = Particles[i].best_position

## pso main loop
for it in range(maxIt):
    for i in range(nPop):
        
        # update velocity
        Particles[i].velocity = w*Particles[i].velocity  \
            + c1*np.random.rand(1,nVar)[0]*(Particles[i].best_position - Particles[i].position) \
            + c2*np.random.rand(1,nVar)[0]*(globalBest_position - Particles[i].position)
              
        # update position
        Particles[i].position = Particles[i].position + Particles[i].velocity 

        # evaluation
        Particles[i].cost = costfunction(Particles[i].position)

        # update personal best
        if  Particles[i].cost <  Particles[i].best_cost:
            Particles[i].best_position = Particles[i].position
            Particles[i].best_cost = Particles[i].cost

            # update global best
            if Particles[i].best_cost < globalBest_cost:
                globalBest_cost = Particles[i].best_cost
                globalBest_position = Particles[i].best_position 

    
 
    w=w*wdamping
    BestCostMemory = np.append([BestCostMemory], [globalBest_cost])
    print(globalBest_cost)
    BestPosionMemory = np.append(BestPosionMemory , [globalBest_position],axis=0)
    print(globalBest_position)



plt.plot(BestCostMemory) 
plt.show()



