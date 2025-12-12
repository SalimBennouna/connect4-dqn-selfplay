import numpy as np
import gym
import gym_connect4  # noqa: F401
import torch

from connect4_dqn.models.q_network import QNetwork
from connect4_dqn.agents.policy import Policy
from connect4_dqn.env.wrappers import obs_to_flat_state


def test_policy_action_is_legal_move_no_epsilon():
    env = gym.make("Connect4-v0", height=6, width=9, connect=4)
    obs = env.reset()
    state = obs_to_flat_state(obs)

    n_obs = env.height * env.width
    n_act = env.width
    qnet = QNetwork(n_obs, n_act, l1=16, l2=16)

    # Force deterministic preference: make column 0 have highest q value
    with torch.no_grad():
        # last linear layer is at the end of Sequential; easiest is set all weights to 0 then bias for action 0
        last = None
        for m in qnet.layers:
            if isinstance(m, torch.nn.Linear):
                last = m
        assert last is not None
        last.weight.zero_()
        last.bias.zero_()
        last.bias[0] = 1.0  # make action 0 best

    policy = Policy(env, qnet, epsilon=0.0)  # no exploration

    action = policy(state, no_epsilon=True)
    assert action in env.get_moves()

    env.close()


def test_policy_action_is_random_legal_when_epsilon_one():
    env = gym.make("Connect4-v0", height=6, width=9, connect=4)
    obs = env.reset()
    state = obs_to_flat_state(obs)

    n_obs = env.height * env.width
    n_act = env.width
    qnet = QNetwork(n_obs, n_act, l1=16, l2=16)

    policy = Policy(env, qnet, epsilon=1.0)  # always explore

    for _ in range(20):
        a = policy(state, no_epsilon=False)
        assert a in env.get_moves()

    env.close()