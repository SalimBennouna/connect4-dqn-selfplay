import itertools
import torch
import numpy as np

from ..env.wrappers import better_step, obs_to_flat_state


def train_naive_agent_against(
    env,
    optimizer,
    trainee,
    opponent,
    loss_fn,
    lr_scheduler,
    num_episodes,
    gamma,
    device,
):
    rewards = []

    for _ in range(num_episodes):
        obs = env.reset()
        state = obs_to_flat_state(obs)

        player = np.random.choice([0, 1])
        if player == 1:
            a2 = opponent(state)
            state, _, _ = better_step(env, a2, player)

        total_r = 0
        for _ in itertools.count():
            if not env.get_moves():
                break

            q = trainee.qnetwork(torch.tensor(state, device=device))
            action = trainee(state)

            next_state, _, done = better_step(env, action, 1 - player)
            reward = 1 if done else 0

            with torch.no_grad():
                nq = trainee.qnetwork(torch.tensor(next_state, device=device)).max()
                target = reward + gamma * nq

            loss = loss_fn(q[action], target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_r += reward
            if done:
                break
            state = next_state

        trainee.decay_epsilon()
        rewards.append(total_r)

    return rewards