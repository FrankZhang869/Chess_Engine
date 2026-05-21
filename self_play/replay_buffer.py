"""
replay_buffer.py
================
Stores self-play game positions and samples from them for training.

Each entry in the buffer is a (state, policy, value) tuple where:
    state:   18-plane board tensor (18, 8, 8) int8
    policy:  sparse policy — (indices, probs) arrays of length MULTIPV
    value:   game outcome from current player's perspective (+1/0/-1)

The buffer is a fixed-size deque — oldest entries are automatically
discarded when the buffer is full. This keeps training data fresh
and prevents early low-quality games from dominating training forever.
"""

import numpy as np
from collections import deque
import pickle
from pathlib import Path


# Maximum number of positions stored across all games
# 20 iterations × 200 games × 40 moves = 160,000 positions
MAX_BUFFER_SIZE = 40,000

# Number of top moves stored per position (sparse policy)
MULTIPV = 5

# Path to save/load the buffer between Colab sessions
BUFFER_PATH = "replay_buffer.pkl"


class ReplayBuffer:
    """
    Fixed-size circular buffer of self-play positions.

    Stores positions as raw numpy arrays and converts to tensors
    only when sampling for training — saves memory during generation.
    """

    def __init__(self, max_size=MAX_BUFFER_SIZE):
        self.max_size = max_size
        # Each element: (state, policy_indices, policy_probs, value)
        self.buffer   = deque(maxlen=max_size)

    def __len__(self):
        return len(self.buffer)

    def add_game(self, states, policy_indices, policy_probs, values):
        """
        Add all positions from one self-play game to the buffer.

        Args:
            states:          list of (18,8,8) int8 arrays
            policy_indices:  list of (MULTIPV,) int16 arrays
            policy_probs:    list of (MULTIPV,) float32 arrays
            values:          list of floats — game outcome from each player's perspective
        """
        for s, pi, pp, v in zip(states, policy_indices, policy_probs, values):
            self.buffer.append((s, pi, pp, float(v)))

    def sample(self, batch_size):
        """
        Randomly sample batch_size positions from the buffer.

        Returns:
            states:          (batch, 18, 8, 8) float32
            policy_indices:  (batch, MULTIPV)  int64
            policy_probs:    (batch, MULTIPV)  float32
            values:          (batch,)           float32
        """
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch   = [self.buffer[i] for i in indices]

        states         = np.stack([b[0] for b in batch]).astype(np.float32)
        policy_indices = np.stack([b[1] for b in batch]).astype(np.int64)
        policy_probs   = np.stack([b[2] for b in batch]).astype(np.float32)
        values         = np.array([b[3] for b in batch],  dtype=np.float32)

        return states, policy_indices, policy_probs, values

    def save(self, path=BUFFER_PATH):
        """Save buffer to disk so it persists between Colab sessions."""
        with open(path, "wb") as f:
            pickle.dump(self.buffer, f)
        print(f"Replay buffer saved: {len(self.buffer):,} positions → {path}")

    def load(self, path=BUFFER_PATH):
        """Load buffer from disk if it exists."""
        if Path(path).exists():
            with open(path, "rb") as f:
                self.buffer = pickle.load(f)
            print(f"Replay buffer loaded: {len(self.buffer):,} positions from {path}")
        else:
            print(f"No existing buffer found at {path} — starting fresh.")