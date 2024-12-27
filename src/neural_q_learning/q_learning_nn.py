import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from grid_world import GridWorld


def one_hot(state):
    HEIGHT, WIDTH = 3, 4
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    y, x = state
    idx = y * WIDTH + x
    vec[idx] = 1
    return vec[np.newaxis, :]


class QNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = 4

        self.q_net = QNet(12, self.action_size)
        self.optimizer = optim.SGD(self.q_net.parameters(), lr=self.lr)
        self.optimizer.zero_grad()

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            qs = self.q_net(state)
            return torch.argmax(qs).item()

    def update_q(self, state, action, reward, next_state, done):
        if done:
            next_q = np.zeros(1)
        else:
            next_qs = self.q_net(next_state)
            next_q = torch.max(next_qs).item()

        target = reward + self.gamma * next_q
        qs = self.q_net(state)
        q = qs[:, action]
        loss = F.mse_loss(q, torch.tensor(target, dtype=torch.float32))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


if __name__ == "__main__":
    # q_net = QNet(12, 4)
    # state = (2, 3)
    # state = one_hot(state)
    # state = torch.tensor(state, dtype=torch.float32)
    # q_s = q_net(state)
    # print(q_s.shape)

    env = GridWorld()
    agent = QLearningAgent()

    episodes = 1000
    loss_history = []

    for ep in range(episodes):
        state = env.reset()
        state = one_hot(state)
        state = torch.tensor(state, dtype=torch.float32)

        total_loss, count = 0, 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = one_hot(next_state)
            next_state = torch.tensor(next_state, dtype=torch.float32)

            loss = agent.update_q(state, action, reward, next_state, done)
            total_loss += loss
            count += 1

            state = next_state

        average_loss = total_loss / count
        loss_history.append(average_loss)
        print(f"Episode: {ep}, Loss: {average_loss}")

    plt.plot(loss_history)
    plt.show()
