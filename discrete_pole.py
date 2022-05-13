import gym
import math
import numpy as np
import sklearn
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env.reset()

episodes = 1000
gamma = 1.0
eps = .9
alpha = 1.0

# replace technically legal [-inf,inf] with reasonable bounds on values
max_state = env.observation_space.high
max_state[1] = 0.5
max_state[-1] = math.radians(50) / 1.
min_state = env.observation_space.low
min_state[1] = -0.5
min_state[-1] = -math.radians(50) / 1.

num_state_buckets = [1,1,6,12]
state_deltas = [(max_state[i] - min_state[i]) / num_state_buckets[i] for i in range(len(max_state))]

def get_discrete_state(state):
    return [
        max(
            min(
                int((state[i] - min_state[i]) / state_deltas[i]), 
                int((max_state[i] - min_state[i]) / state_deltas[i]) - 1
            ), 
        0) for i in range(len(state))]

def eps_greedy(epsilon, state, Q):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    s1, s2, s3, s4 = get_discrete_state(state)
    return np.argmax(Q[s1, s2, s3, s4, :])

Q = np.zeros((num_state_buckets + [env.action_space.n]))
episode_lens = []

for episode in range(episodes):
    if episode > 0 and episode % 200 == 0:
        eps /= 2
        alpha /= 2
        print(f"episode : {episode}")

    done = False
    state = env.reset()

    t = 0
    while not done:
        action = eps_greedy(eps, state, Q)
        new_state, reward, done, _ = env.step(action)
        
        s1, s2, s3, s4 = get_discrete_state(state)
        new_s1, new_s2, new_s3, new_s4 = get_discrete_state(new_state)
        
        td_delta = gamma * np.max(Q[new_s1, new_s2, new_s3, new_s4, :]) + reward \
            - Q[s1, s2, s3, s4, action]
        Q[s1, s2, s3, s4, action] += alpha * td_delta
        
        state = new_state
        t += 1

    episode_lens.append(t)

# final test
state = env.reset()
done = False
while not done:
    env.render()
    state, _, done, _ = env.step(eps_greedy(0, state, Q))
env.close()

plt.plot(range(len(episode_lens)), episode_lens)
plt.show()