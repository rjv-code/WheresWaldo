"""
q_agent.py — Q-learning agent for RL-guided Waldo tile selection.

The agent maintains a Q-table (dictionary) mapping (state, action) pairs
to expected cumulative rewards. It uses epsilon-greedy action selection:
  - With probability ε  → pick a random unchecked tile (explore)
  - With probability 1−ε → pick the tile with the highest Q-value (exploit)

Epsilon decays over training so the agent explores early and exploits later.

"""

from __future__ import annotations

import pickle
import random
from collections import defaultdict
from pathlib import Path


class QAgent:


    def __init__(
        self,
        n_tiles: int,
        alpha: float       = 0.1,
        gamma: float       = 0.9,
        epsilon: float     = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.n_tiles       = n_tiles
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: (state_tuple, action_int) → float
        # defaultdict means unseen (state, action) pairs start at 0.0
        self.Q: dict[tuple, float] = defaultdict(float)

    # Action selection

    def select_action(
        self, state: tuple[int, ...], valid_actions: list[int]
    ) -> int:
        """
        Epsilon-greedy action selection.

        Always picks from valid_actions (unchecked tiles only).
        With probability ε → random choice (explore).
        Otherwise → tile with highest Q-value (exploit).
        """
        if not valid_actions:
            raise ValueError("No valid actions — all tiles already checked.")

        if random.random() < self.epsilon:
            # Explore: random unchecked tile
            return random.choice(valid_actions)

        # Exploit: tile with highest Q-value among valid actions
        q_values = {a: self.Q[(state, a)] for a in valid_actions}
        max_q    = max(q_values.values())

        # Break ties randomly so the agent doesn't always prefer low indices
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    
    # Q-table update

    def update(
        self,
        state:        tuple[int, ...],
        action:       int,
        reward:       float,
        next_state:   tuple[int, ...],
        next_valid:   list[int],
    ) -> None:
        """
        Standard Q-learning (off-policy) Bellman update:

            Q(s, a) ← Q(s, a) + α * [ r + γ * max_a' Q(s', a') − Q(s, a) ]

        If the episode is done (next_valid is empty), the future term is 0.
        """
        current_q = self.Q[(state, action)]

        if next_valid:
            future_q = max(self.Q[(next_state, a)] for a in next_valid)
        else:
            future_q = 0.0  # terminal state — no future reward

        target         = reward + self.gamma * future_q
        self.Q[(state, action)] = current_q + self.alpha * (target - current_q)

    # Epsilon decay

    def decay_epsilon(self) -> None:
        """Call once per episode after the update loop."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # Persistence

    def save(self, path: str = "qtable.pkl") -> None:
        """Save the Q-table and hyperparameters to disk."""
        payload = {
            "Q":             dict(self.Q),
            "n_tiles":       self.n_tiles,
            "alpha":         self.alpha,
            "gamma":         self.gamma,
            "epsilon":       self.epsilon,
            "epsilon_min":   self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"  Q-table saved → {path}  ({len(self.Q)} entries)")

    def load(self, path: str = "qtable.pkl") -> None:
        """Load a previously saved Q-table from disk."""
        if not Path(path).exists():
            raise FileNotFoundError(f"No Q-table found at {path}")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.Q             = defaultdict(float, payload["Q"])
        self.n_tiles       = payload["n_tiles"]
        self.alpha         = payload["alpha"]
        self.gamma         = payload["gamma"]
        self.epsilon       = payload["epsilon"]
        self.epsilon_min   = payload["epsilon_min"]
        self.epsilon_decay = payload["epsilon_decay"]
        print(f"  Q-table loaded ← {path}  ({len(self.Q)} entries)")

    
    # Diagnostics

    def stats(self) -> dict:
        """Return a snapshot of agent state for logging."""
        return {
            "epsilon":    round(self.epsilon, 4),
            "q_entries":  len(self.Q),
            "q_min":      round(min(self.Q.values(), default=0), 3),
            "q_max":      round(max(self.Q.values(), default=0), 3),
        }
