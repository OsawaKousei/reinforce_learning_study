import copy

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
from torch import nn, optim


class QNet(nn.Module):
    def __init__(self, action_size: int):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 2

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.q_net = QNet(self.action_size)
        self.q_net_target = QNet(self.action_size)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.optimizer.zero_grad()

    def sync_q_net(self) -> None:
        self.q_net_target = copy.deepcopy(self.q_net)

    def get_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = state[np.newaxis, :]
            qs: torch.Tensor = self.q_net(torch.tensor(state, dtype=torch.float32))
            arg = qs.argmax().item()

            assert isinstance(arg, int)
            return arg

    def update(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        state = torch.tensor(state, dtype=torch.float32)
        qs = self.q_net(state)
        q = qs[np.arange(self.batch_size), action]

        next_state = torch.tensor(next_state, dtype=torch.float32)
        next_qs = self.q_net_target(next_state)
        next_q = next_qs.max(dim=1).values

        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        target = reward + self.gamma * next_q * (1 - done)

        loss = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    episodes = 300
    sync_interval = 20
    env = gym.make("CartPole-v1", render_mode="human")
    agent = DQNAgent()
    reward_history = []

    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.update(
                state,
                action,
                reward,
                next_state,
                done,
            )
            state = next_state
            total_reward += reward

        if episode % sync_interval == 0:
            agent.sync_q_net()

        reward_history.append(total_reward)
        print(f"Episode: {episode}, Reward: {total_reward}")

    plt.plot(reward_history)
    plt.show()

    agent.epsilon = 0.0
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        total_reward += reward
        env.render()

    print(f"Test reward: {total_reward}")
