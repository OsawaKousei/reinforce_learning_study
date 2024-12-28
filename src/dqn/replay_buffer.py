import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int):
        self.buffer: deque = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self) -> int:
        return len(self.buffer)

    def get_batch(self) -> tuple:
        data = random.sample(self.buffer, self.batch_size)
        state = np.stack([d[0] for d in data])
        action = np.array([d[1] for d in data])
        reward = np.array([d[2] for d in data])
        next_state = np.stack([d[3] for d in data])
        done = np.array([d[4] for d in data])
        return state, action, reward, next_state, done
