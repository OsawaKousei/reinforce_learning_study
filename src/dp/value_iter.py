from collections import defaultdict

from grid_world import GridWorld


def value_iter_one_step(
    V: defaultdict, env: GridWorld, gamma: float = 0.9
) -> defaultdict:
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_values = {}

        for action in env.actions():
            next_state = env.next_state(state, action)
            reward = env.reward(state, action, next_state)
            value = reward + gamma * V[next_state]
            action_values[action] = value

        V[state] = max(action_values.values())

    return V


def value_iter(
    V: defaultdict,
    env: GridWorld,
    gamma: float = 0.9,
    threshold: float = 1e-3,
    is_render: bool = False,
) -> defaultdict:
    while True:
        if is_render:
            env.render_v(V)

        old_V = V.copy()
        V = value_iter_one_step(V, env, gamma)

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        if delta < threshold:
            break

    return V


if __name__ == "__main__":
    env = GridWorld()
    V: defaultdict[str, float] = defaultdict(lambda: 0)

    V = value_iter(V, env, is_render=True)
    env.render_v(V)
