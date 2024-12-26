import matplotlib.pyplot as plt
import numpy as np


class NonStatBandit:
    def __init__(self, arms: int = 10):
        self.rates = np.random.rand(arms)

    def play(self, arm: int):
        rate = self.rates[arm]
        self.rates += 0.01 * np.random.randn(len(self.rates))
        if rate > np.random.rand():
            return 1
        else:
            return 0


class AlphaAgent:
    def __init__(self, epsilon: float, alpha: float, action_size: int = 10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        else:
            return np.argmax(self.Qs)


if __name__ == "__main__":
    steps = 1000
    epsilon = 0.1

    bandit = NonStatBandit()
    agent = AlphaAgent(epsilon, 0.8)

    total_reward = 0
    total_rewards = []
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward

        total_rewards.append(total_reward)
        rates.append(total_reward / (step + 1))

    print("Total reward:", total_reward)

    plt.ylabel("Total reward")
    plt.xlabel("Steps")
    plt.plot(total_rewards)
    plt.show()

    plt.ylabel("Rates")
    plt.xlabel("Steps")
    plt.plot(rates)
    plt.show()
