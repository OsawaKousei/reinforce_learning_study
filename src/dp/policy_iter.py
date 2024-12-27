from collections import defaultdict

from grid_world import GridWorld
from policy_eval import policy_eval


def argmax(d: dict) -> int:
    max_value = max(d.values())
    max_key = 0
    for k, v in d.items():
        if v == max_value:
            max_key = k
    return max_key


def greedy_policy(V: defaultdict, env: GridWorld, gamma: float = 0.9) -> defaultdict:
    pi: defaultdict = defaultdict(dict)

    for state in env.states():
        action_values = {}

        for action in env.actions():
            next_state = env.next_state(state, action)
            reward = env.reward(state, action, next_state)
            value = reward + gamma * V[next_state]
            action_values[action] = value
            max_action = argmax(action_values)
            action_probs = {0: 0, 1: 0, 2: 0, 3: 0}
            action_probs[max_action] = 1
            pi[state] = action_probs

    return pi


def policy_iter(
    pi: defaultdict,
    V: defaultdict,
    env: GridWorld,
    gamma: float = 0.9,
    threshold: float = 1e-3,
    is_render: bool = False,
) -> defaultdict:
    while True:
        V = policy_eval(pi, V, env, gamma, threshold)
        new_pi = greedy_policy(V, env, gamma)

        if is_render:
            env.render_v(V, new_pi)

        if pi == new_pi:
            break

        pi = new_pi

    return pi


if __name__ == "__main__":
    env = GridWorld()
    V: defaultdict[str, float] = defaultdict(lambda: 0)
    pi: defaultdict[str, dict[str, float]] = defaultdict(
        lambda: dict.fromkeys(env.actions(), 0.25)
    )
    pi = policy_iter(pi, V, env, is_render=True)
    env.render_v(V, pi)
