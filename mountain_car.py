import gym
import numpy as np
import matplotlib.pyplot as plt

# for continuous (non-discretized) state space
import sklearn
from sklearn.linear_model import SGDRegressor
import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler


discrete = True

env = gym.make('MountainCar-v0')
env.reset()

# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to convert a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))

def featurize_state(state):
    scaled = scaler.transform([state])
    featurized = featurizer.transform(scaled)
    return featurized[0]


max_state = env.observation_space.high
min_state = env.observation_space.low

if discrete:
    state_space_size = 2
    state_buckets = [20] * state_space_size
    state_deltas = [(max_state[i] - min_state[i]) / state_buckets[i] for i in range(state_space_size)]

episodes = 1000
alpha = 0.1
epsilon = 0.9
gamma = 0.95

if discrete:
    Q = np.zeros(state_buckets + [env.action_space.n])
else:
    state = env.reset()

    Q = [SGDRegressor() for _ in range(env.action_space.n)]
    Q = [q.partial_fit([featurize_state(state)], [0]) for q in Q]

def discretize_state(state):
    return [int(state[i] / state_deltas[i]) for i in range(state_space_size)]

def eps_greedy_action(Q, state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()

    if discrete:
        s1, s2 = discretize_state(state)
        return np.argmax(Q[s1, s2, :])
    else:
        return np.argmax([q.predict([featurize_state(state)]) for q in Q])

rewards = []

for episode in range(episodes):
    if episode > 0 and episode % 100 == 0:
        print(episode)
        epsilon /= 2

    state = env.reset()
    action = eps_greedy_action(Q, state, epsilon)
    done = False

    episode_reward = 0
    while not done:
        if episode % 50 == 0:
            env.render()

        new_state, reward, done, _ = env.step(action)
        new_action = eps_greedy_action(Q, new_state, epsilon)

        if discrete:
            s1, s2 = discretize_state(state)
            new_s1, new_s2 = discretize_state(new_state)

            td_target = reward + gamma * Q[new_s1, new_s2, new_action] - Q[s1, s2, action]
            Q[s1, s2, action] += alpha * td_target
        else:
            td_target = reward + gamma * Q[new_action].predict([featurize_state(new_state)])
            Q[action] = Q[action].partial_fit([featurize_state(state)], td_target)

        state = new_state
        action = new_action

        episode_reward += reward
    rewards.append(episode_reward)

# final "evaluation" run
done = False
state = env.reset()
while not done:
    env.render()
    action = eps_greedy_action(Q, state, 0)
    state, _, done, _ = env.step(action)

plt.plot(range(len(rewards)), rewards)
plt.show()