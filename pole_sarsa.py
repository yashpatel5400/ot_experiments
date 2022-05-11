import gym
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

env = gym.make('CartPole-v0')
state = env.reset()

episodes = 1000
epsilon = 0.5
gamma = 0.99
alpha = 0.9

def eps_greedy_choose(Qs, state, eps):
	if np.random.random() < eps:
		return int(np.random.random() * env.action_space.n)
	return np.argmax([Q.predict([state]) for Q in Qs])

Qs = [SGDRegressor() for _ in range(env.action_space.n)]
[Q.partial_fit([state], [0]) for Q in Qs]

episode_lens = []

for episode in range(episodes):
	if episode % 100 == 0 and episode > 0:
		print(episode)
		epsilon /= 2

	state = env.reset()
	done = False

	t = 0

	while not done:
		if episode % 50 == 0:
			env.render()

		action = eps_greedy_choose(Qs, state, epsilon)
		new_state, reward, done, info = env.step(action)
		new_action = eps_greedy_choose(Qs, new_state, epsilon)

		td_delta = (reward + gamma * Qs[new_action].predict([new_state])) - Qs[action].predict([state])
		td_target = Qs[action].predict([state]) + alpha * td_delta
		Qs[action] = Qs[action].partial_fit([state], np.array([td_target]).ravel())

		state = new_state
		action = new_action
		t += 1

	episode_lens.append(t)

plt.plot(range(len(episode_lens)), episode_lens)
plt.show()