import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PolicyNet(nn.Module):
    def __init__(self, action_size: int = 2):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x


class ValueNet(nn.Module):
    def __init__(self) -> None:
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Agent:
    def __init__(self) -> None:
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005
        self.action_size = 2

        self.pi = PolicyNet(self.action_size)
        self.v = ValueNet()
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)
        self.optimizer_pi.zero_grad()
        self.optimizer_v.zero_grad()

    def get_action(self, state: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        state = state[np.newaxis, :]
        probs = self.pi(torch.tensor(state, dtype=torch.float32))
        probs = probs[0]
        m = torch.distributions.Categorical(probs)
        action = m.sample().item()
        return action, probs[action]

    def update(
        self,
        state: np.ndarray,
        action_prob: torch.Tensor,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]

        target = reward + self.gamma * self.v(
            torch.tensor(next_state, dtype=torch.float32)
        ) * (1 - done)
        target.detach()
        v = self.v(torch.tensor(state, dtype=torch.float32))
        loss_v = F.mse_loss(v, target)

        delta = target - v
        loss_pi = -torch.log(action_prob) * delta

        self.optimizer_v.zero_grad()
        self.optimizer_pi.zero_grad()

        (loss_pi + loss_v).backward()

        self.optimizer_v.step()
        self.optimizer_pi.step()


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")
    agent = Agent()
    episodes = 2000
    reward_history = []

    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0

        while not done:
            action, prob = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.update(state, prob, reward, next_state, done)

            state = next_state
            total_reward += reward

        reward_history.append(total_reward)
        if episode % 100 == 0:
            print("episode :{}, total reward : {:.1f}".format(episode, total_reward))
    env.close()
    plt.plot(reward_history)
    plt.show()
