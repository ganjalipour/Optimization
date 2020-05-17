import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
import random as random
import math

def deleteOneRepositoryMember(rep , gamma):
    gridindices = [item.gridIndex for item in rep]
    OCells = np.unique(gridindices) # ocupied cells
    N = np.zeros(len(OCells))
    for k in range(len(OCells)):
        N[k] = gridindices.count(OCells[k])
    # selection probablity
    p = [math.exp(gamma*item) for item in N]
    p = np.array(p)/sum(p)

    # select cell index
    sci = roulettewheelSelection(p)
    SelectedCell = OCells[sci]

    #selected Cell members
    selectedCellmembers = [item for item in gridindices if item == SelectedCell]

    selectedmemberindex = np.random.randint(0,len(selectedCellmembers))
    #selectedmember = selectedCellmembers[selectedmemberindex]

    # delete memeber
    #rep[selectedmemberindex] = []
    rep = np.delete(rep, selectedmemberindex)

    return rep.tolist()


def SelectLeader(rep , beta):
    gridindices = [item.gridIndex for item in rep]
    OCells = np.unique(gridindices) # ocupied cells
    N = np.zeros(len(OCells))
    for k in range(len(OCells)):
        N[k] = gridindices.count(OCells[k])
    # selection probablity
    p = [math.exp(-beta*item) for item in N]
    p = np.array(p)/sum(p)

    # select cell index
    sci = roulettewheelSelection(p)
    SelectedCell = OCells[sci]

    #selected Cell members
    selectedCellmembers = [item for item in gridindices if item == SelectedCell]

    selectedmemberindex = np.random.randint(0,len(selectedCellmembers))
    # selectedmember = selectedCellmembers[selectedmemberindex]

    return rep[selectedmemberindex]



def roulettewheelSelection(p):
    r = random.random()
    cumsum = np.cumsum(p)
    y = (cumsum<r)
    x= [i for i in y if i==True]
    return len(x)

def FindGridIndex(particle, grid):
    nObj = len(particle.cost)
    NGrid = len(grid[0].LowerBounds)
    
    particle.gridSubIndex = np.zeros((1,nObj))[0]
    for j in range(nObj):  
        index_in_Dim = len( [item for item in grid[j].UpperBounds if particle.cost[j]>item]) 
        particle.gridSubIndex[j] = index_in_Dim

    particle.gridIndex = particle.gridSubIndex[0]

    for j in range(1,nObj):
        particle.gridIndex = particle.gridIndex 
        particle.gridIndex = NGrid*particle.gridIndex
        particle.gridIndex = particle.gridIndex + particle.gridSubIndex[j]

    return particle



def CreateGrid(pop,nGrid,alpha,nobj):
    costs = [item.cost for item in pop]
    Cmin = np.min(costs,axis=0)
    Cmax = np.max(costs,axis=0)
    deltaC = Cmax - Cmin
    Cmin =  Cmin - alpha*deltaC
    Cmax = Cmax + alpha*deltaC
   
    grid = [GridDim() for p in range(nobj)]
    for i in range(nobj):
       dimValues = np.linspace(Cmin[i],Cmax[i],nGrid+1).tolist()
       grid[i].LowerBounds = [-float('inf')] + dimValues
       grid[i].UpperBounds = dimValues  + [float('inf')]
    return grid



def Dominates(x,y):
    x=np.array(x)
    y=np.array(y)
    x_dominate_y = all(x<=y) and any(x<y)
    return x_dominate_y

def DetermineDomination(pop):
    pop_len= len(pop)
    for i in range(pop_len):
         pop[i].IsDominated = False 

    for i in range(pop_len-1):
        for j in range(i+1,pop_len):
            if Dominates(pop[i].cost,pop[j].cost):
                pop[j].IsDominated = True
            if Dominates(pop[j].cost,pop[i].cost):
                pop[i].IsDominated = True

    return pop

        
# problem definition
def MOP2(x):
    x = np.array(x)
    n= len(x)
    z1 = 1 - math.exp(-sum((x-1/math.sqrt(n))**2))
    z2 = 1 - math.exp(-sum((x+1/math.sqrt(n))**2))
    return [z1,z2]

costfunction = lambda x: MOP2(x)

nVar = 5 # number of decision vars
varMin = -4
varMax = 4
maxIt = 100
nPop = 200    # population size
nRep = 50  # size of repository
w = 0.5 # inertia wieght
c1 = 2 # personal learning coefficient
c2 = 2 # global learning coefficient
wdamping = 0.99


# ################ constriction coefficients
# phi1 = 2.05
# phi2 = 2.05
# phi = phi1+phi2
# chi = 2/(phi - 2 + np.sqrt(phi**2 - 4*phi))
# w = chi # inertia wieght
# c1 = chi*phi1 # personal learning coefficient
# c2 = chi*phi2 # global learning coefficient
# wdamping = 1
# #################

beta = 1 # leader selection pressure
gamma = 1 # deletion selection pressure
NoGrid = 5
alpha=0.1 # nerkhe tavarrom grid

# initialization
class Particle:
    position = []
    cost = []
    velocity = []
    best_position = []
    best_cost = []
    IsDominated = []
    gridIndex = []
    gridSubIndex = []

# for each objective a grid items is division of values of objective cost
class GridDim: 
    LowerBounds = []
    UpperBounds = []

#Particles = np.matlib.repmat(Particle,nPop,1)
Particles = [Particle() for p in range(nPop)]
for i in range(nPop):
    Particles[i].position = np.random.uniform(varMin,varMax,nVar)
    Particles[i].velocity = np.zeros(nVar)
    Particles[i].cost = costfunction(Particles[i].position)
    # update best personal Best
    Particles[i].best_position = Particles[i].position
    Particles[i].best_cost = Particles[i].cost
    Particles[i].IsDominated = False 

Particles = DetermineDomination(Particles)

Repos = [item for item in Particles if item.IsDominated == False ]
nObj =len( Repos[0].cost)
grid = CreateGrid(Repos,NoGrid,alpha=0.1,nobj=nObj)

for r in range(len(Repos)):
    Repos[r] = FindGridIndex(Repos[0],grid)

# MOPSO main loop
for it in range(maxIt):
    for i in range(nPop):
        leader = SelectLeader(Repos,beta)
        # update velocity
        Particles[i].velocity = w*Particles[i].velocity  \
            + c1*np.random.rand(1,nVar)[0]*(Particles[i].best_position - Particles[i].position) \
            + c2*np.random.rand(1,nVar)[0]*(leader.position - Particles[i].position)
              
        # update position
        Particles[i].position = Particles[i].position + Particles[i].velocity 

        # evaluation
        Particles[i].cost = costfunction(Particles[i].position)

        if Dominates(Particles[i].cost,Particles[i].best_cost):
            Particles[i].best_position = Particles[i].position
            Particles[i].best_cost = Particles[i].cost
        else:
            if np.random.rand() > 0.5:
                Particles[i].best_position = Particles[i].position
                Particles[i].best_cost = Particles[i].cost
      

    Repos = Repos + Particles
    Repos = DetermineDomination(Repos)
    Repos = [item for item in Repos if item.IsDominated == False ]

    grid = CreateGrid(Repos,NoGrid,alpha=0.1,nobj=nObj)
    for r in range(len(Repos)):
        Repos[r] = FindGridIndex(Repos[r],grid)

    # check if repository is full
    if len(Repos) > nRep :
        extra = len(Repos) - nRep
        for e in range(extra):
            Repos = deleteOneRepositoryMember(Repos,gamma)

    ########## show figure ########## 
    plt.clf()
    particlesCost = np.reshape( [item.cost for item in Particles ],newshape=(nPop,2))
    repositoryCost = [item.cost for item in Repos]
    repositoryCost = np.reshape( repositoryCost, newshape=(len(repositoryCost),2))
    plt.plot(particlesCost[:,0], particlesCost[:,1], 'o' ,mfc='none')
    plt.plot(repositoryCost[:,0], repositoryCost[:,1], 'r*')
    
    plt.draw()
    plt.pause(0.00000000001)

    w=w*wdamping
    
    # print(repositoryCost)
    # print("ok")
    # print(particlesCost)
    ########## show figure ##########
plt.show()









