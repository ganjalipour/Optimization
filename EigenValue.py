import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import time
#from scipy.sparse import sparse

n = 1000
P = np.random.rand(n,n)
print(P)

S = P @ P.T
print(S)



def BinverseA(A,B):
    Binverse = la.inv(B)
    return np.dot(Binverse,A)


#############################################################
from scipy.sparse import spdiags

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



dataB = np.array([[a0]]).reshape((1,n))
diagsB = np.array([0])
datatestOstadB = spdiags(data, diags, n, n).toarray()

BinvA = BinverseA(datatestOstad,datatestOstadB)
print("BinverseA :")
print(BinvA)

evalues, evectors = la.eig(BinvA)
print("eigenValues: ")
print(evalues)
print("eignenVecors:")
print(evectors)

# Atest = np.array([[3,-1,-2],[-1,1,3],[-2,3,5]])
# Atest = np.array([[0.5,-1.375,0.125],[-1.5,0.375,-0.125],[1.5,1.375,1.875]])

# B = np.random.rand(n,n)
# BPDef = np.dot(B,B.T)

# BPDefInerse = la.inv(BPDef)
# Atest2 = np.dot(BPDefInerse, Atest)/100
# print(" A , B  are :")
# print(Atest2)
# print(Atest)
# Import Keras backend
# import keras.backend as K
# Define SMAPE loss function
#############################################################
def dlambda_dt(vin, Ain, lam):
    v=np.array( vin)
    A= np.array(Ain)
    AV = np.dot(A, v)
    LamV = lam * v
    e = AV - LamV
    result = - np.dot(e,v.T)
    return(result)

def dV_dt(vin, Ain, lam, k):
    v= np.array(vin)
    A= np.array(Ain)
    AV = np.dot(A, v)
    LamV = lam * v
    e = AV - LamV
    eA = np.dot(e, A)
    lame = lam * e
    vvT = np.dot(v,v.T)
    kvvvT=k*v*(vvT-1)
    result= eA - lame +kvvvT
    return result


def GradiantDescentOptimizer(eigenvalue,vector,InputA,m,k,iteration):
    print(vector)
    counter = 0

    diagramValue= np.array([eigenvalue])
    diagramVector = np.array((n,1))

    while(counter <  iteration):
        print(eigenvalue)
        print(vector)
        eigenvalue = -m*dlambda_dt( vector,Ain=InputA,lam=eigenvalue) + eigenvalue
        vector = -m*dV_dt(vector,Ain=InputA,lam= eigenvalue,k= k) + vector
        diagramValue =np.append( [diagramValue] , [eigenvalue])
        counter = counter + 1

    plt.plot(diagramValue)
    plt.show()


def MomentumOptimizer(eigenvalue,vector,InputA,alpha,beta,k,iteration):
    print(vector)
    counter = 0

    diagramValue= np.array([eigenvalue])
    diagramVector = np.array((n,1))
    v1 = 0
    v2 = 0

    while(counter <  iteration):
        print(eigenvalue)
        print(vector)
        eigenvalue0=eigenvalue
        v1 = beta * v1 + (1-beta ) *dlambda_dt( vector,Ain=InputA,lam=eigenvalue)
        eigenvalue = eigenvalue - alpha * v1

        v2 = beta * v2 + (1-beta) * dV_dt(vector,Ain=InputA,lam= eigenvalue,k= k)
        vector = vector - alpha * v2

        diagramValue =np.append( [diagramValue] , [eigenvalue])
        counter = counter + 1

        # if(eigenvalue-eigenvalue0<0.0000001):
        #     break

    plt.plot(diagramValue)
    plt.show()



# iteration =200
# counter = 0
# m = 0.01
# k = 20
#vector = [ 0 , 0 , 1]
vector = [0 ,0, 0, 0, 0, 0 ,0, 0 ,0 ,0.2]*100
vector = np.random.uniform(0,2,1000).tolist()
#GradiantDescentOptimizer(eigenvalue=0, vector=vector, InputA=BinvA, m=0.01, k=20, iteration=500)
MomentumOptimizer(eigenvalue=0.5,vector=vector,InputA=BinvA,alpha=0.11,beta=0.7,k=20,iteration=200)


# print(vector)

# eigenvalue=0

# diagramValue= np.array([eigenvalue])
# diagramVector = np.array((n,1))

# while(counter <  iteration):
#     print(eigenvalue)
#     print(vector)
#     eigenvalue = m*dlambda_dt( vector,Ain=datatestOstad,lam=eigenvalue) + eigenvalue
#     vector = -m*dV_dt(vector,Ain=datatestOstad,lam= eigenvalue,k= k) + vector
#     diagramValue =np.append( [diagramValue] , [eigenvalue])
#     counter =counter+1

# plt.plot(diagramValue)
# plt.show()



# def customLoss(xin , Ain):
#     epsilon = 0.1
#     x= np.array(xin)
#     A= np.array(Ain)
#     I = np.identity(A.shape[0])
#     L1 = np.dot(np.dot(x.T,x), A)
#     L2 =  np.dot(np.dot(x.T,A), x)
#     L3= 1 - L2
#     L4 = np.dot( L3 , I)
#     L5 = L1  + L4
#     L6 = np.dot(L5,x)
#     dx_dt = -x + L6
#     return dx_dt

# iteration =32   
# counter = 0
# xin=[ 0.33279072 , 0.65186476, -0.53590413, -0.42086747]
# while(counter <  iteration):
#     xin = customLoss( xin,Ain=S)
#     print(xin)
#     counter+=counter






