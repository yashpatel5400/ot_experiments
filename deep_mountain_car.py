import gym
import random
import copy
import numpy as np
import matplotlib.pyplot as plt

# Deep RL specific packages
import torch
import torch.nn as nn
import torch.optim as optim

# for continuous (non-discretized) state space
import sklearn
from sklearn.linear_model import SGDRegressor
import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler

env = gym.make('MountainCar-v0')
env.reset()

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.relu = nn.ReLU()
        self.d1 = nn.Linear(env.observation_space.shape[0], 16)
        self.d2 = nn.Linear(16, 64)
        self.d3 = nn.Linear(64, env.action_space.n)

    def forward(self, x):
        x = torch.from_numpy(x)
        x = self.d1(x)
        x = self.relu(x)
        x = self.d2(x)
        x = self.relu(x)
        x = self.d3(x)
        return x

batch_size = 25
episodes = 1000
alpha = 0.1
epsilon = 0.9
gamma = 0.95

state = env.reset()
policy_model = DQN()
train_model = DQN()

def eps_greedy_action(model, state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    return np.argmax(model(state).detach().numpy())

rewards = []
criterion = nn.MSELoss()
optimizer = optim.Adam(train_model.parameters())

for episode in range(episodes):
    if episode > 0 and episode % 100 == 0:
        policy_model = copy.deepcopy(train_model)
        epsilon /= 2

    state = env.reset()
    action = eps_greedy_action(policy_model, state, epsilon)
    done = False

    episode_reward = 0

    replay_buffer = []
    while not done:
        if episode % 50 == 0:
            env.render()

        new_state, reward, done, _ = env.step(action)
        new_action = eps_greedy_action(policy_model, new_state, epsilon)

        td_target = reward + gamma * policy_model(new_state)[new_action]
        replay_buffer.append((state, action, td_target.detach().numpy()))

        state = new_state
        action = new_action

        episode_reward += reward

    random.shuffle(replay_buffer)
    training_batch = replay_buffer[:batch_size]
    states, actions, Qs = list(map(list, zip(*training_batch)))
    states = np.array(states)
    Qs = torch.from_numpy(np.array(Qs))
    
    idxs = torch.from_numpy(np.array(actions)).long().unsqueeze(1)
    Qhat = train_model(states).gather(1, idxs).squeeze()

    optimizer.zero_grad()
    loss = criterion(Qhat, Qs)
    loss.backward()
    optimizer.step()

    rewards.append(episode_reward)

# final "evaluation" run
done = False
state = env.reset()
while not done:
    env.render()
    action = eps_greedy_action(train_model, state, 0)
    state, _, done, _ = env.step(action)

plt.plot(range(len(rewards)), rewards)
plt.show()