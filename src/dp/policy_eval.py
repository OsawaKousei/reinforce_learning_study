from collections import defaultdict

from grid_world import GridWorld


def eval_one_step(
    pi: defaultdict, V: defaultdict, env: GridWorld, gamma: float = 0.9
) -> defaultdict:
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_probs = pi[state]
        new_V = 0

        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            reward = env.reward(state, action, next_state)
            new_V += action_prob * (reward + gamma * V[next_state])

        V[state] = new_V

    return V


def policy_eval(
    pi: defaultdict,
    V: defaultdict,
    env: GridWorld,
    gamma: float = 0.9,
    threshold: float = 1e-3,
) -> defaultdict:
    while True:
        old_V = V.copy()
        V = eval_one_step(pi, V, env, gamma)

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
    pi: defaultdict[str, dict[str, float]] = defaultdict(
        lambda: dict.fromkeys(env.actions(), 0.25)
    )

    V = policy_eval(pi, V, env)
    env.render_v(V)
