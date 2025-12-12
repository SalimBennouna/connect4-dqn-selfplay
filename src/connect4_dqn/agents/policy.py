import random
import numpy as np
import torch


class Policy:
    """
    Epsilon-greedy policy over a Q-network.
    Expects env to provide `get_moves()` returning legal column indices.
    """

    def __init__(self, env, qnetwork, epsilon=0.5, epsilon_min=0.013, epsilon_decay=0.9875):
        self.env = env
        self.qnetwork = qnetwork
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

    def __call__(self, state: np.ndarray, no_epsilon: bool = False) -> int:
        legal_moves = self.env.get_moves()
        if len(legal_moves) == 0:
            raise ValueError("No legal moves available.")

        # Forward pass
        with torch.no_grad():
            q_values = self.qnetwork(torch.tensor(state, dtype=torch.float32))

        # Explore
        if (not no_epsilon) and (random.random() < self.epsilon):
            return int(random.choice(legal_moves))

        # Exploit: best legal move
        best_move = int(legal_moves[0])
        best_q = q_values[best_move].item()

        for m in legal_moves:
            m = int(m)
            v = q_values[m].item()
            if v > best_q:
                best_q = v
                best_move = m

        return best_move

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_epsilon(self, epsilon: float = 0.5) -> None:
        self.epsilon = float(epsilon)