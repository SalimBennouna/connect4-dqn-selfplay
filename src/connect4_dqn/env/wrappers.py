import numpy as np


def obs_to_flat_state(obs) -> np.ndarray:
    return np.asarray(obs[0][1]).flatten()


def better_step(env, action: int, player: int):
    observed_space, reward_vector, winner, info = env.step(action)

    states = observed_space[player]
    player_states = states[1]
    opponent_states = states[2]

    done = bool(winner) or len(info["legal_actions"]) == 0
    trad_state = player_states - opponent_states

    return trad_state.flatten(), float(reward_vector[player]), done