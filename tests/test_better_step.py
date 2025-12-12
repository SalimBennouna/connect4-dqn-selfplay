import numpy as np
import gym
import gym_connect4  # noqa: F401

from connect4_dqn.env.wrappers import better_step


def test_better_step_returns_expected_types():
    env = gym.make("Connect4-v0", height=6, width=9, connect=4)
    obs = env.reset()

    # choose a legal move
    legal = env.get_moves()
    assert len(legal) > 0
    action = int(legal[0])

    # pick a player index (0 or 1). better_step expects you pass the player you want perspective for.
    player = 0

    state, reward, done = better_step(env, action, player)

    assert isinstance(state, np.ndarray)
    assert state.ndim == 1
    assert state.shape[0] == env.height * env.width

    assert isinstance(reward, float)
    assert isinstance(done, (bool, np.bool_))

    env.close()


def test_better_step_done_when_no_legal_actions_or_winner():
    env = gym.make("Connect4-v0", height=4, width=4, connect=4)  # smaller board to finish quickly
    env.reset()

    # play until termination or move limit
    player = 0
    for _ in range(100):
        legal = env.get_moves()
        if not legal:
            break
        action = int(legal[0])
        _, _, done = better_step(env, action, player)
        if done:
            break

    # by now either ended or ran out of moves
    # done can be False if we didn't finish within 100 moves, but on 4x4 connect4 it should terminate fast
    assert done is True

    env.close()