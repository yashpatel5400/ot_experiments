import gym
import numpy as np
import sklearn
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler

env = gym.make('CartPole-v0')
env.reset()

episodes = 1000
gamma = 0.1
eps = 0.1

# construct feature extractor
observation_examples = []
for _ in range(2500):
    state, _, done, _ = env.step(env.action_space.sample())
    print(state)
    if done:
    	env.reset()
    observation_examples.append(state)

observation_examples = np.array([observation_examples]).squeeze()
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to convert a state to a featurized representation.
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


def eps_greedy(epsilon, state, Qs):
	if np.random.random() < epsilon:
		return env.action_space.sample()
	pred_Qs = [Q.predict([featurize_state(state)]) for Q in Qs]
	return np.argmax(pred_Qs)

# sorta hacky, but having one Q per A allows us to very directly find the best action
# by just iterating over the A and comparing Q(s). in cont action spaces, we would optimize over A
state = env.reset()
Qs = [SGDRegressor() for _ in range(env.action_space.n)]
[Q.partial_fit([featurize_state(state)], [0]) for Q in Qs]

for _ in range(episodes):
	env.render() # no render on train
	done = False

	state = env.reset()
	action = eps_greedy(eps, state, Qs)

	while not done:
		new_state, reward, done, _ = env.step(action)
		new_action = eps_greedy(eps, new_state, Qs)
		
		td_target = gamma * Qs[new_action].predict([featurize_state(new_state)]) + reward \
			- Qs[action].predict([featurize_state(state)])
		Qs[action].partial_fit([featurize_state(state)], [td_target])
		
		state = new_state
		action = new_action

# final test
state = env.reset()
while not done:
	env.render()
	state, _, _, _ = env.step(eps_greedy(0, state, Qs))
env.close()