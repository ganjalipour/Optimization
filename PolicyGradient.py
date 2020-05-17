from keras.layers import Dense, Activation, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import keras.backend as k 
import numpy as np 

class Agent(object):
    def __init__(self,alpha,gamma=0.99, NActions=3,layer1size=16,
    layer2size=16,inputdims=128,fname='reinfoce.h5'):
        self.gamma =gamma
        self.lr = alpha
        self.inputdims = inputdims
        self.fc1dims = layer1size
        self.fc2dims = layer2size
        self.NoActions = NActions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        

        self.policy, self.predict = self.build_policy_network()
        self.action_space= [i for i in range(NActions)]
        self.model_file = fname

    def build_policy_network(self):
        input = Input(shape=(self.inputdims,))
        advantages = Input(shape=[1])
        dense1 = Dense(self.fc1dims,activation='relu')(input)
        dense2 = Dense(self.fc2dims,activation='relu')(dense1)
        probs = Dense(self.NoActions, activation='softmax')(dense2)

        def CutomLoss(yTrue,ypred):
            out = k.clip(ypred,1e-8,1-1e-8)
            logLikelyhood = yTrue * k.log(out)
            return k.sum(-logLikelyhood*advantages)
        
        policy = Model(input=[input,advantages],output=[probs])
        policy.compile(optimizer=Adam(lr=self.lr),loss=CutomLoss)

        predict = Model(input=[input],output= [probs])

        return policy, predict

    def choose_action(self,observation):
        state = observation[np.newaxis, :]
        probablities = self.predict.predict(state)[0]
        action = np.random.choice(self.action_space,p=probablities)
        return action

    def store_transition(self,observation,action,reward):
        self.action_memory.append(action)
        self.state_memory.append(observation)
        self.reward_memory.append(reward)

    def learn(self):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)

        actions = np.zeros([len(action_memory), self.NoActions])
        actions[np.arange(len(action_memory)),action_memory] = 1

        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k]*discount
                discount *=self.gamma

            G[t] = G_sum

        mean = np.mean(G)
        std = np.std(G) if np.std(G)>0 else 1
        self.G = (G-mean) / std

        cost = self.policy.train_on_batch([state_memory, self.G],actions)
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

    def savemodel(self):
        self.policy.save(self.model_file)

    def loadmodel(self):
        self.policy = load_model(self.model_file)


import gym
import matplotlib.pyplot as plt
#from utils import plotlearning

agent = Agent(alpha=0.0005, inputdims=2,gamma=0.99, NActions=3, layer1size=64,layer2size=64)

env = gym.make('MountainCar-v0').env
scoreHistory = []
n_episodes = 2000

for i in range(n_episodes):
    done = False
    score = 0
    observation =  env.reset()
    while not done:
        action = agent.choose_action(observation)
        observ , reward,done,info = env.step(action)
        agent.store_transition(observation, action, reward)
        observation = observ
        score += reward

    scoreHistory.append(score)
    agent.learn()
    print('episode  ',i, 'score ',score)

plt.plot(scoreHistory)
plt.show()


    





        

