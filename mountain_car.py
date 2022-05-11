import gym
import numpy as np

env = gym.make('MountainCar-v0')
env.reset()

max_state = env.observation_space.high
min_state = env.observation_space.low

state_space_size = 2
state_buckets = [10] * state_space_size
state_deltas = [(max_state[i] - min_state[i]) / state_buckets[i] for i in range(state_space_size)]

episodes = 1000
alpha = 0.1
epsilon = 0.9
gamma = 0.9

Q = np.zeros(state_buckets + [env.action_space.n])

def discretize_state(state):
    return [int(state[i] / state_deltas[i]) for i in range(state_space_size)]

def eps_greedy_action(Q, state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    s1, s2 = discretize_state(state)
    return np.argmax(Q[s1, s2, :])

for episode in range(episodes):
    if episode > 0 and episode % 100 == 0:
        epsilon /= 2

    state = env.reset()
    action = eps_greedy_action(Q, state, epsilon)
    done = False

    while not done:
        if episode % 50 == 0:
            print(episode)
            env.render()

        new_state, reward, done, _ = env.step(action)
        new_action = eps_greedy_action(Q, new_state, epsilon)

        s1, s2 = discretize_state(state)
        new_s1, new_s2 = discretize_state(new_state)

        td_target = reward + gamma * Q[new_s1, new_s2, new_action] - Q[s1, s2, action]
        Q[s1, s2, action] += alpha * td_target

        state = new_state
        action = new_action