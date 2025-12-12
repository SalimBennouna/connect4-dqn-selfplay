#!/usr/bin/env python3
import argparse
import os
import random
from dataclasses import dataclass
from typing import Callable, Deque, List, Tuple
import collections
import itertools

import numpy as np
import torch
import torch.nn.init as init

import yaml
import gym
import gym_connect4  # noqa: F401


# -----------------------------
# Model / Policy (from notebook)
# -----------------------------
class QNetwork(torch.nn.Module):
    def __init__(self, n_observations: int, n_actions: int,
                 nn_l1: int, nn_l2: int, nn_l3=0, nn_l4=0, nn_l5=0):
        super().__init__()
        if nn_l3 == 0:
            nn_l3 = nn_l2
        if nn_l4 == 0:
            nn_l4 = nn_l3
        if nn_l5 == 0:
            nn_l5 = nn_l4

        self.layer1 = torch.nn.Linear(n_observations, nn_l1)
        self.layer2 = torch.nn.Linear(nn_l1, nn_l2)
        self.layer3 = torch.nn.Linear(nn_l2, nn_l3)
        self.layer4 = torch.nn.Linear(nn_l3, nn_l4)
        self.layer5 = torch.nn.Linear(nn_l4, nn_l5)
        self.layer6 = torch.nn.Linear(nn_l5, n_actions)
        self.act = torch.nn.LeakyReLU()

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6]:
            init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.layer1.weight.dtype)
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        x = self.act(self.layer3(x))
        x = self.act(self.layer4(x))
        x = self.act(self.layer5(x))
        return self.layer6(x)


class Policy:
    def __init__(self, env, qnetwork: QNetwork,
                 epsilon=0.5, epsilon_min=0.013, epsilon_decay=0.9875):
        self.env = env
        self.qnetwork = qnetwork
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

    def __call__(self, state: np.ndarray, no_epsilon: bool = False) -> int:
        available_moves = self.env.get_moves()
        q_values = self.qnetwork(torch.tensor(state))
        if (random.random() < self.epsilon) and (not no_epsilon):
            return int(random.choice(available_moves))

        best_move = int(available_moves[0])
        best_q_value = q_values[best_move]
        for move in available_moves:
            if q_values[move] > best_q_value:
                best_move = int(move)
                best_q_value = q_values[move]
        return best_move

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_epsilon(self, epsilon: float = 0.5) -> None:
        self.epsilon = float(epsilon)


def better_step(env, action: int, player: int):
    observed_space, reward_vector, winner, info = env.step(action)
    states = observed_space[player]
    player_states = states[1]
    opponent_states = states[2]
    done = bool(winner) or (len(info["legal_actions"]) == 0)
    trad_state = player_states - opponent_states
    return trad_state.flatten(), float(reward_vector[player]), done


class MinimumExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, lr_decay: float, min_lr: float = 1e-6):
        self.min_lr = float(min_lr)
        super().__init__(optimizer, gamma=float(lr_decay), last_epoch=-1)

    def get_lr(self):
        return [max(base_lr * self.gamma ** self.last_epoch, self.min_lr) for base_lr in self.base_lrs]


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer = collections.deque(maxlen=self.capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, int(action), float(reward), next_state, bool(done)))

    def sample(self, batch_size: int):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, int(batch_size)))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)


def train_dqn_replay(env, first_agent: Policy, opponent: Policy,
                     optimizer, loss_fn: Callable, lr_scheduler,
                     num_episodes: int, gamma: float,
                     replay_buffer: ReplayBuffer, batch_size: int,
                     device: torch.device) -> List[float]:
    episode_rewards: List[float] = []

    for _episode in range(1, int(num_episodes) + 1):
        obs = env.reset()
        state = obs[0][1].flatten()

        player = int(np.random.choice([0, 1]))
        if player == 1 and len(env.get_moves()) > 0:
            a2 = opponent(state)
            state, _, _ = better_step(env, a2, player)

        total_r = 0.0
        for _t in itertools.count():
            if len(env.get_moves()) == 0:
                done = True
                reward = 0.0
                next_state = state
            else:
                action = first_agent(state)
                next_state, done, reward = better_step(env, action, 1 - player)
                if (not done) and (len(env.get_moves()) > 0):
                    a2 = opponent(next_state)
                    next_state, reward2, done2 = better_step(env, a2, player)
                    reward = -1.0 * reward2
                    done = done2

            replay_buffer.add(state, action, reward, next_state, done)

            if len(replay_buffer) >= int(batch_size):
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                states_t = torch.tensor(states, device=device)
                actions_t = torch.tensor(actions, device=device)
                rewards_t = torch.tensor(rewards, device=device)
                next_states_t = torch.tensor(next_states, device=device)
                dones_t = torch.tensor(dones, device=device)

                q_values = first_agent.qnetwork(states_t)
                next_q_values = first_agent.qnetwork(next_states_t)
                target = rewards_t + float(gamma) * torch.max(next_q_values, dim=1).values * (~dones_t)

                pred = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
                loss = loss_fn(pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

            total_r += float(reward)
            if done:
                break
            state = next_state

        episode_rewards.append(total_r)
        first_agent.decay_epsilon()

    return episode_rewards


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(cfg) -> gym.Env:
    return gym.make(cfg["env"]["id"],
                    height=cfg["env"]["height"],
                    width=cfg["env"]["width"],
                    connect=cfg["env"]["connect"])


def save_checkpoint(path: str, qnet: QNetwork, cfg: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "state_dict": qnet.state_dict(),
        "cfg": cfg,
    }
    torch.save(payload, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None, help="override checkpoint path")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 0)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(cfg)

    n_obs = env.width * env.height
    n_act = env.width

    mcfg = cfg["model"]
    qnet = QNetwork(n_obs, n_act,
                    mcfg["hidden_l1"], mcfg["hidden_l2"],
                    mcfg.get("hidden_l3", 0), mcfg.get("hidden_l4", 0), mcfg.get("hidden_l5", 0)).to(device)

    agent_cfg = cfg["agent"]
    first_agent = Policy(env, qnet,
                         epsilon=agent_cfg["epsilon_start"],
                         epsilon_min=agent_cfg["epsilon_min"],
                         epsilon_decay=agent_cfg["epsilon_decay"])

    opponent_qnet = QNetwork(n_obs, n_act,
                             mcfg["hidden_l1"], mcfg["hidden_l2"],
                             mcfg.get("hidden_l3", 0), mcfg.get("hidden_l4", 0), mcfg.get("hidden_l5", 0)).to(device)
    opponent_qnet.load_state_dict(qnet.state_dict())
    opponent_agent = Policy(env, opponent_qnet,
                            epsilon=agent_cfg["epsilon_start"],
                            epsilon_min=agent_cfg["epsilon_min"],
                            epsilon_decay=agent_cfg["epsilon_decay"])

    tcfg = cfg["train"]
    optimizer = torch.optim.AdamW(first_agent.qnetwork.parameters(), lr=float(tcfg["lr"]), amsgrad=True)
    lr_scheduler = MinimumExponentialLR(optimizer, lr_decay=float(tcfg["lr_decay"]), min_lr=float(tcfg["min_lr"]))
    loss_fn = torch.nn.MSELoss()
    replay_buffer = ReplayBuffer(int(tcfg["replay_buffer_capacity"]))

    num_trainings = int(tcfg["num_trainings"])
    opp_every = int(tcfg["opponent_update_every"])
    save_every = int(cfg["output"].get("save_every_trainings", 50))
    run_name = cfg["output"].get("run_name", "run")
    ckpt_dir = cfg["output"].get("checkpoints_dir", "checkpoints")

    results_dir = cfg["output"].get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    for k in range(num_trainings):
        # reset epsilon at start of each training block
        first_agent.reset_epsilon(agent_cfg["epsilon_start"])

        _rewards = train_dqn_replay(
            env=env,
            first_agent=first_agent,
            opponent=opponent_agent,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            num_episodes=int(tcfg["num_episodes"]),
            gamma=float(tcfg["gamma"]),
            replay_buffer=replay_buffer,
            batch_size=int(tcfg["batch_size"]),
            device=device,
        )

        if (k % opp_every) == 0:
            opponent_agent.qnetwork.load_state_dict(first_agent.qnetwork.state_dict())

        if (k % save_every) == 0 or (k == num_trainings - 1):
            out_path = args.out or os.path.join(ckpt_dir, f"{run_name}_trainings{k:04d}.pth")
            save_checkpoint(out_path, first_agent.qnetwork, cfg)
            print(f"[saved] {out_path}")

    env.close()


if __name__ == "__main__":
    main()