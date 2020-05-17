import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 

class GenericNetwork(nn.Module):
    def __init__(self,lr,input_dims,fc1_dims,fc2_dims,n_actions):
        super(GenericNetwork,self).__init__()
        self.lr = lr
        self.inputDims = input_dims
        self.fc1Dims = fc2_dims
        self.fc2Dims = fc2_dims
        self.nActions = n_actions
        
        self.fc1 = nn.Linear(*self.inputDims,self.fc1Dims)
        self.fc2 = nn.Linear(self.fc1Dims,self.fc2Dims)
        self.fc3 = nn.Linear(self.fc2Dims,self.nActions)

        self.optimizer = optim.Adam(self.parameters(), lr= self.lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu' )
        self.to(self.device)

    def forward(self,observation):
        state = T.tensor(observation, dtype= T.float).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent(object):
    def __init__(self, alpha, beta, inputDims, gamma= 0.99,nActions=2,
                  layer1size=64,layer2size=64, n_outputs= 1):
        self.gamma = gamma
        self.log_probs = None
        self.n_outputs = n_outputs
        
        self.actor = GenericNetwork(alpha, inputDims,layer1size,layer2size,nActions)
        self.critic = GenericNetwork(beta, inputDims, layer1size,layer2size,1)

    def chooseAction(self, observation):
        mu, sigma = self.actor.forward(observation)
        sigma = T.exp(sigma)
        action_probs = T.distributions.Normal(mu,sigma)
        probs = action_probs.sample(sample_shape=T.Size([self.n_outputs]))
        self.log_probs = action_probs.log_prob(probs).to(self.actor.device)
        action = T.tanh(probs)
        return action.item()

    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        critic_value_next = self.critic.forward(new_state)
        critic_value = self.critic.forward(state)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        delta = reward + self.gamma*critic_value_next*(1-int(done)) - critic_value

        actor_loss = -self.log_probs*delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()


import gym
import matplotlib as matplot

agent = Agent(alpha = 0.0005, beta = 0.0001, inputDims=[2],gamma=0.99,
layer1size=256, layer2size=256)
env = gym.make('MountainCarContinuous-v0')
score_history = []
num_episodes = 100
for i in range(num_episodes):
    done = False
    score = 0
    observation = env.reset()
    while not done:
        action = np.array(agent.chooseAction(observation)).reshape((1,))
        observationNext, reward, done , info = env.step(action)
        agent.learn(observation,reward,observationNext,done)
        observation= observationNext
        score += reward
    score_history.append(score)
    print('episode ',i,'score ', score)



    





