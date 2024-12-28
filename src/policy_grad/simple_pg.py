import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Policy(nn.Module):
    def __init__(self, action_size: int):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x


class Agent:
    def __init__(self) -> None:
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2

        self.memory = []
        self.pi = Policy(self.action_size)
        self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)
        self.optimizer.zero_grad()

    def get_action(self, state: np.ndarray) -> tuple:
        state = state[np.newaxis, :]
        state = torch.tensor(state, dtype=torch.float32)
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.detach().numpy())
        return action, probs[action]

    def add(self, reward: float, prob: torch.Tensor) -> None:
        data = (reward, prob)
        self.memory.append(data)

    def update(self) -> None:
        self.pi.zero_grad()
        G, loss = torch.tensor(0.0), torch.tensor(0.0)
        for reward, _ in reversed(self.memory):
            G = self.gamma * G + reward

        for _, prob in self.memory:
            loss += -torch.log(prob) * G

        loss.backward()
        self.optimizer.step()
        self.memory = []


if __name__ == "__main__":
    # env = gym.make("CartPole-v1", render_mode="human")
    # state = env.reset()[0]
    # agent = Agent()

    # action, prob = agent.get_action(state)
    # print("action: ", action, "prob: ", prob)

    # G = 100.0
    # J = G * torch.log(prob)
    # print("J: ", J)

    # J.backward()

    episodes = 3000
    env = gym.make("CartPole-v1", render_mode="human")
    agent = Agent()
    reward_history = []

    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0

        while not done:
            action, prob = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.add(reward, prob)
            state = next_state
            total_reward += reward

        agent.update()
        reward_history.append(total_reward)
        print(f"Episode: {episode}, Total Reward: {total_reward}")
