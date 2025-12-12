import itertools
import numpy as np

from ..env.wrappers import better_step, obs_to_flat_state


def play_game(env, agent, opponent, no_eps_agent=False, no_eps_opp=False):
    obs = env.reset()
    state = obs_to_flat_state(obs)

    player = np.random.choice([0, 1])
    if player == 1:
        a2 = opponent(state, no_eps_opp)
        state, _, _ = better_step(env, a2, player)

    for _ in itertools.count():
        if not env.get_moves():
            return 0

        a = agent(state, no_eps_agent)
        next_state, _, done = better_step(env, a, 1 - player)
        if done:
            return 1

        a2 = opponent(next_state, no_eps_opp)
        next_state, _, done = better_step(env, a2, player)
        if done:
            return -1

        state = next_state