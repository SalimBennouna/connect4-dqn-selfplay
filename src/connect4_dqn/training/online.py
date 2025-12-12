import itertools
import torch
import numpy as np

from ..env.wrappers import better_step, obs_to_flat_state


def play_game_with_train(
    env,
    agent,
    opponent,
    optimizer,
    loss_fn,
    gamma,
    lr_scheduler,
    replay_buffer,
    batch_size,
    device,
):
    obs = env.reset()
    state = obs_to_flat_state(obs)

    player = np.random.choice([0, 1])
    if player == 1:
        a2 = opponent(state)
        state, _, _ = better_step(env, a2, player)

    for _ in itertools.count():
        if not env.get_moves():
            return 0

        action = agent(state)
        next_state, _, done = better_step(env, action, 1 - player)
        reward = 1 if done else 0

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

        if done:
            return reward

        state = next_state