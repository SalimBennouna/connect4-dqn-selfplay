import itertools
import torch
import numpy as np

from ..env.wrappers import better_step, obs_to_flat_state


def train_dqn_replay(
    env,
    agent,
    opponent,
    optimizer,
    loss_fn,
    lr_scheduler,
    num_episodes,
    gamma,
    replay_buffer,
    batch_size,
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

            action = agent(state)
            next_state, done, reward = better_step(env, action, 1 - player)

            if not done and env.get_moves():
                a2 = opponent(next_state)
                next_state, reward2, done = better_step(env, a2, player)
                reward = -reward2

            replay_buffer.add(state, action, reward, next_state, done)

            if len(replay_buffer) >= batch_size:
                s, a, r, ns, d = replay_buffer.sample(batch_size)

                s = torch.tensor(s, device=device)
                a = torch.tensor(a, device=device)
                r = torch.tensor(r, device=device)
                ns = torch.tensor(ns, device=device)
                d = torch.tensor(d, device=device)

                q = agent.qnetwork(s).gather(1, a.unsqueeze(1)).squeeze()
                nq = agent.qnetwork(ns).max(1).values
                target = r + gamma * nq * (~d)

                loss = loss_fn(q, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

            total_r += reward
            if done:
                break
            state = next_state

        agent.decay_epsilon()
        rewards.append(total_r)

    return rewards