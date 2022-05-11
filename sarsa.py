import gym
import numpy as np
import sklearn
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler

env = gym.make('CartPole-v0')
env.reset()

episodes = 1000
gamma = 1.0
eps = 1.0


def eps_greedy(epsilon, state, Qs):
	if np.random.random() < epsilon:
		return env.action_space.sample()
	pred_Qs = [Q.predict([state]) for Q in Qs]
	return np.argmax(pred_Qs)

# sorta hacky, but having one Q per A allows us to very directly find the best action
# by just iterating over the A and comparing Q(s). in cont action spaces, we would optimize over A
state = env.reset()
Qs = [SGDRegressor(learning_rate="constant") for _ in range(env.action_space.n)]
[Q.partial_fit([state], [0]) for Q in Qs]

for episode in range(episodes):
	if episode > 0 and episode % 100 == 0:
		eps /= 2
		print(f"epsilon={eps}")

	env.render() # no render on train
	done = False

	state = env.reset()
	action = eps_greedy(eps, state, Qs)

	while not done:
		new_state, reward, done, _ = env.step(action)
		new_action = eps_greedy(eps, new_state, Qs)
		
		td_target = gamma * Qs[new_action].predict([new_state]) + reward \
			- Qs[action].predict([state])
		Qs[action] = Qs[action].partial_fit([state], np.array([td_target]).ravel())
		
		state = new_state
		action = new_action

# final test
state = env.reset()
while not done:
	env.render()
	state, _, _, _ = env.step(eps_greedy(0, state, Qs))
env.close()