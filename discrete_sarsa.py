import gym
import numpy as np
import sklearn
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler

env = gym.make('CartPole-v0')
env.reset()

episodes = 1000
gamma = 0.9
eps = 1.0
alpha = 0.9

# replace technically legal [-inf,inf] with reasonable bounds on values
max_state = env.observation_space.high
max_state[1] = 4
max_state[-1] = 4
min_state = env.observation_space.low
min_state[1] = -4
min_state[-1] = -4

num_state_buckets = [5,5,5,5]
state_deltas = [(max_state[i] - min_state[i]) / num_state_buckets[i] for i in range(len(max_state))]

def get_discrete_state(state):
	return [int((state[i] - min_state[i]) / state_deltas[i]) for i in range(len(state))]

def eps_greedy(epsilon, state, Q):
	if np.random.random() < epsilon:
		return env.action_space.sample()
	s1, s2, s3, s4 = get_discrete_state(state)
	return np.argmax(Q[s1, s2, s3, s4, :])

Q = np.zeros((num_state_buckets + [env.action_space.n]))

for episode in range(episodes):
	if episode > 0 and episode % 100 == 0:
		eps /= 2
		print(f"epsilon={eps}")

	done = False

	state = env.reset()
	action = eps_greedy(eps, state, Q)

	while not done:
		if episode % 50 == 0:
			env.render()
			
		new_state, reward, done, _ = env.step(action)
		new_action = eps_greedy(eps, new_state, Q)
		
		s1, s2, s3, s4 = get_discrete_state(state)
		new_s1, new_s2, new_s3, new_s4 = get_discrete_state(new_state)

		td_target = gamma * Q[new_s1, new_s2, new_s3, new_s4, new_action] + reward \
			- Q[s1, s2, s3, s4, action]
		Q[s1, s2, s3, s4, action] += alpha * td_target
		
		state = new_state
		action = new_action

# final test
state = env.reset()
while not done:
	env.render()
	state, _, _, _ = env.step(eps_greedy(0, state, Q))
env.close()