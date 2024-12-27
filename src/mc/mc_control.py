import numpy as np


def greedy_probs(Q, state, epsilon: float = 0, action_size: int = 4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)

    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}
    action_probs[max_action] += 1 - epsilon

    return action_probs
