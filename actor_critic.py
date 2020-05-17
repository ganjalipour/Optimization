from keras import backend as k
from keras.layers import   Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

class Agent(object):
    def __init__(self,alpha,beta,gamma=0.99, N_actions=4,
                   lalyer1Size=1024,layer2Size=512,inputDims=8):
        self.gamma = gamma
        self.alpha =alpha
        self.beta =beta
        self.inputDims = inputDims
        self.fc1Dims = lalyer1Size
        self.fc2Dims = layer2Size
        self.n_actions = N_actions
        self.actor, self.critic, self.policy = self.buildNetwork()
        self.action_Space = [i for i in range(self.n_actions)]

    def buildNetwork(self):
        input = Input(shape=(self.inputDims,))
        delta = Input(shape=[1])
        dense1 = Dense(self.fc1Dims, activation='relu')(input)
        dense2 = Dense(self.fc2Dims,activation='relu')(dense1)
        probs = Dense(self.n_actions, activation='softmax')(dense2)
        values = Dense(1,activation='linear')(dense2)

        def customLoss(yTrue,yPred):
            out = k.clip(yPred, 1e-8,1-1e-8)
            logLikelyhood = yTrue* k.log(out)
            return k.sum(-logLikelyhood*delta)
        
        actor = Model(input = [input,delta], output=[probs])
        actor.compile(optimizer=Adam(learning_rate=self.alpha),loss=customLoss)
        
        critic = Model(input = [input],output =[values])
        critic.compile(optimizer= Adam(learning_rate=self.beta),loss='mean_squared_error')

        policy = Model(input=[input], output=[probs])

        return actor, critic, policy
    
    def chooseAction(self,observation):
        state = observation[np.newaxis, :]
        probablities = self.policy.predict(state)[0]
        action = np.random.choice(self.action_Space, p=probablities)
        return action

    def learn(self, state, action , reward, nextstate,done):
        state = state[np.newaxis, :]
        nextstate = nextstate[np.newaxis, :]
        criticValueNext = self.critic.predict(nextstate)
        criticValue = self.critic.predict(state)

        target = reward + self.gamma*criticValueNext*(1-int(done))
        delta = target - criticValue

        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1

        self.actor.fit([state,delta],actions, verbose=0)
        self.critic.fit(state,target, verbose=0)


import matplotlib.pyplot as plt
#from utils import plotlearning
import gym

agent = Agent(alpha = 0.0001,beta = 0.0005,gamma=0.99, N_actions=3,lalyer1Size=1024,layer2Size=512,inputDims=2)

env = gym.make('MountainCar-v0').env
scoreHistory = []
n_episodes = 2000

for i in range(n_episodes):
    done = False
    score = 0
    observation =  env.reset()
    while not done:
        action = agent.chooseAction(observation)
        nextobservation , reward,done,info = env.step(action)
        agent.learn(nextobservation, action, reward,nextobservation,done)
        observation = nextobservation
        score += reward

    scoreHistory.append(score)
    
    print('episode  ',i, 'score ',score)

plt.plot(scoreHistory)
plt.show()

