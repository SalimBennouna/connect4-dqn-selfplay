#!/usr/bin/env python3
import argparse
import os
import random
import collections
import itertools

import numpy as np
import torch
import torch.nn.init as init
import yaml
import gym
import gym_connect4  # noqa: F401

import matplotlib.pyplot as plt


class QNetwork(torch.nn.Module):
    def __init__(self, n_observations, n_actions, nn_l1, nn_l2, nn_l3=0, nn_l4=0, nn_l5=0):
        super().__init__()
        if nn_l3 == 0: nn_l3 = nn_l2
        if nn_l4 == 0: nn_l4 = nn_l3
        if nn_l5 == 0: nn_l5 = nn_l4
        self.l1 = torch.nn.Linear(n_observations, nn_l1)
        self.l2 = torch.nn.Linear(nn_l1, nn_l2)
        self.l3 = torch.nn.Linear(nn_l2, nn_l3)
        self.l4 = torch.nn.Linear(nn_l3, nn_l4)
        self.l5 = torch.nn.Linear(nn_l4, nn_l5)
        self.l6 = torch.nn.Linear(nn_l5, n_actions)
        self.act = torch.nn.LeakyReLU()
        for layer in [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6]:
            init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = x.to(self.l1.weight.dtype)
        x = self.act(self.l1(x)); x = self.act(self.l2(x)); x = self.act(self.l3(x))
        x = self.act(self.l4(x)); x = self.act(self.l5(x))
        return self.l6(x)

class Policy:
    def __init__(self, env, qnet, epsilon=0.5, eps_min=0.013, eps_decay=0.9875):
        self.env = env
        self.qnet = qnet
        self.epsilon = float(epsilon)
        self.eps_min = float(eps_min)
        self.eps_decay = float(eps_decay)

    def __call__(self, state, no_epsilon=False):
        moves = self.env.get_moves()
        q = self.qnet(torch.tensor(state))
        if (random.random() < self.epsilon) and (not no_epsilon):
            return int(random.choice(moves))
        best = int(moves[0])
        bestv = q[best]
        for m in moves:
            if q[m] > bestv:
                best = int(m); bestv = q[m]
        return best

def better_step(env, action, player):
    obs, reward_vec, winner, info = env.step(int(action))
    states = obs[player]
    player_states = states[1]
    opponent_states = states[2]
    done = bool(winner) or (len(info["legal_actions"]) == 0)
    trad = (player_states - opponent_states).flatten()
    return trad, float(reward_vec[player]), done

class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=int(capacity))
    def add(self, *x):
        self.buf.append(x)
    def sample(self, bs):
        batch = random.sample(self.buf, int(bs))
        s,a,r,ns,d = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(ns), np.array(d)
    def __len__(self):
        return len(self.buf)

class MinimumExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, opt, lr_decay, min_lr):
        self.min_lr = float(min_lr)
        super().__init__(opt, gamma=float(lr_decay), last_epoch=-1)
    def get_lr(self):
        return [max(b * self.gamma ** self.last_epoch, self.min_lr) for b in self.base_lrs]


def play_game(env, agent: Policy, opp: Policy, no_eps_a=False, no_eps_o=False):
    obs = env.reset()
    state = obs[0][1].flatten()
    player = int(np.random.choice([0, 1]))

    if player == 1 and len(env.get_moves()) > 0:
        a2 = opp(state, no_epsilon=no_eps_o)
        state, _, _ = better_step(env, a2, player)

    for _t in itertools.count():
        if len(env.get_moves()) == 0:
            return 0

        a = agent(state, no_epsilon=no_eps_a)
        next_state, _, done = better_step(env, a, 1 - player)
        if done:
            return 1

        if len(env.get_moves()) == 0:
            return 0

        a2 = opp(next_state, no_epsilon=no_eps_o)
        next_state, _, done = better_step(env, a2, player)
        if done:
            return -1

        state = next_state


def play_game_with_adaptation(env, agent: Policy, opp: Policy,
                              optimizer, loss_fn, lr_sched,
                              gamma: float, replay: ReplayBuffer, batch_size: int, device):
    obs = env.reset()
    state = obs[0][1].flatten()
    player = int(np.random.choice([0, 1]))

    if player == 1 and len(env.get_moves()) > 0:
        a2 = opp(state)
        state, _, _ = better_step(env, a2, player)

    for _t in itertools.count():
        if len(env.get_moves()) == 0:
            return 0

        q = agent.qnet(torch.tensor(state))
        a = agent(state)
        next_state, _, done = better_step(env, a, 1 - player)
        reward = 0.0

        if done:
            reward = 1.0
        else:
            if len(env.get_moves()) == 0:
                done = True
                reward = 0.0
            else:
                a2 = opp(next_state)
                next_state2, _, done2 = better_step(env, a2, player)
                next_state = next_state2
                if done2:
                    reward = -1.0
                done = done2

        replay.add(state, a, reward, next_state, done)

        if len(replay) >= int(batch_size):
            states, actions, rewards, next_states, dones = replay.sample(batch_size)
            states_t = torch.tensor(states, device=device)
            actions_t = torch.tensor(actions, device=device)
            rewards_t = torch.tensor(rewards, device=device)
            next_states_t = torch.tensor(next_states, device=device)
            dones_t = torch.tensor(dones, device=device)

            qvals = agent.qnet(states_t)
            nxt = agent.qnet(next_states_t)
            target = rewards_t + float(gamma) * torch.max(nxt, dim=1).values * (~dones_t)
            pred = qvals.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_sched.step()

        if done:
            return int(np.sign(reward))

        state = next_state


def load_checkpoint(path: str):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj.get("cfg", None)
    # fallback: user saved whole module
    if hasattr(obj, "state_dict"):
        return obj.state_dict(), None
    raise ValueError("Unknown checkpoint format")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(cfg["env"]["id"], height=cfg["env"]["height"], width=cfg["env"]["width"], connect=cfg["env"]["connect"])

    mcfg = cfg["model"]
    n_obs = env.width * env.height
    n_act = env.width

    sd, _meta = load_checkpoint(args.checkpoint)

    q1 = QNetwork(n_obs, n_act, mcfg["hidden_l1"], mcfg["hidden_l2"],
                 mcfg.get("hidden_l3", 0), mcfg.get("hidden_l4", 0), mcfg.get("hidden_l5", 0)).to(device)
    q1.load_state_dict(sd)

    q2 = QNetwork(n_obs, n_act, mcfg["hidden_l1"], mcfg["hidden_l2"],
                 mcfg.get("hidden_l3", 0), mcfg.get("hidden_l4", 0), mcfg.get("hidden_l5", 0)).to(device)

    acfg = cfg["agent"]
    agent = Policy(env, q1, epsilon=acfg["epsilon_start"], eps_min=acfg["epsilon_min"], eps_decay=acfg["epsilon_decay"])
    opp = Policy(env, q2, epsilon=acfg["epsilon_start"], eps_min=acfg["epsilon_min"], eps_decay=acfg["epsilon_decay"])

    games = int(cfg["eval"]["games"])
    warmup = int(cfg["eval"].get("warmup_drop", 0))
    no_eps_a = bool(cfg["eval"].get("no_epsilon_agent", False))
    no_eps_o = bool(cfg["eval"].get("no_epsilon_opponent", False))

    adapt_cfg = cfg["eval"]["adaptation"]
    do_adapt = bool(adapt_cfg.get("enabled", False))

    wr = 1.0
    wrs = []

    if do_adapt:
        opt = torch.optim.AdamW(agent.qnet.parameters(), lr=float(adapt_cfg["lr"]), amsgrad=True)
        sched = MinimumExponentialLR(opt, lr_decay=float(adapt_cfg["lr_decay"]), min_lr=float(adapt_cfg["min_lr"]))
        loss_fn = torch.nn.MSELoss()
        replay = ReplayBuffer(int(adapt_cfg["replay_buffer_capacity"]))
        batch_size = int(adapt_cfg["batch_size"])
        gamma = float(adapt_cfg["gamma"])

    wins = 0
    losses = 0
    draws = 0

    for i in range(games):
        if do_adapt:
            res = play_game_with_adaptation(env, agent, opp, opt, loss_fn, sched, gamma, replay, batch_size, device)
        else:
            res = play_game(env, agent, opp, no_eps_a=no_eps_a, no_eps_o=no_eps_o)

        if res == 1:
            wins += 1
            wr = (wr * i + 1) / (i + 1)
        elif res == -1:
            losses += 1
            wr = (wr * i) / (i + 1)
        else:
            draws += 1

        wrs.append(wr)

    wrs_plot = wrs[warmup:] if warmup < len(wrs) else wrs

    out_dir = cfg["output"].get("results_dir", "results")
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(args.checkpoint))[0]
    png_path = os.path.join(out_dir, f"eval_{base}.png")

    plt.title(f"Winrate ({'adapt' if do_adapt else 'no-adapt'})")
    plt.xlabel("Games")
    plt.ylabel("Winrate")
    plt.plot(np.linspace(1, len(wrs_plot), len(wrs_plot)), wrs_plot)
    plt.tight_layout()
    if bool(cfg["output"].get("plot_png", True)):
        plt.savefig(png_path, dpi=160)
        print(f"[saved] {png_path}")
    else:
        plt.show()

    print(f"Wins={wins} Losses={losses} Draws={draws}")

    env.close()


if __name__ == "__main__":
    main()