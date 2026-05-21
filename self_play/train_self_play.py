"""
train_self_play.py
==================
Trains the network on a mix of self-play games and supervised Lichess data.

Mix schedule:
    Early iterations (1-5):   70% supervised + 30% self-play
    Mid iterations (6-10):    40% supervised + 60% self-play
    Late iterations (11+):    10% supervised + 90% self-play

This schedule prevents catastrophic forgetting early on when self-play
game quality is still low, then gradually transitions to pure self-play
signal as the engine gets stronger.

Loss functions are identical to supervised training:
    Policy: KL divergence against MCTS visit count distribution
    Value:  MSE against game outcome (+1/0/-1)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from model import ChessNet2
from self_play.replay_buffer import ReplayBuffer


# =========================
# Configuration
# =========================

POLICY_SIZE  = 64 * 73    # 4672
BATCH_SIZE   = 256
EPOCHS       = 10         # fewer epochs per iteration than supervised training
LR           = 3e-4       # lower LR than initial supervised training
WEIGHT_DECAY = 3e-4
MULTIPV      = 5

SUPERVISED_NPZ = "dataset.npz"   # your 500k Lichess dataset


# =========================
# Supervised Mix Schedule
# =========================

def get_supervised_fraction(iteration):
    """
    Returns fraction of each batch that comes from supervised data.
    Gradually reduces supervised mixing as self-play improves.
    """
    if iteration <= 5:
        return 0.70
    elif iteration <= 10:
        return 0.40
    else:
        return 0.10


# =========================
# Self-Play Dataset
# =========================

class SelfPlayDataset(Dataset):
    """
    Dataset wrapping the replay buffer for training.
    Reconstructs sparse policy into dense vector on the fly.
    """

    def __init__(self, states, policy_indices, policy_probs, values):
        self.states         = torch.tensor(states,         dtype=torch.float32)
        self.policy_indices = torch.tensor(policy_indices, dtype=torch.long)
        self.policy_probs   = torch.tensor(policy_probs,   dtype=torch.float32)
        self.values         = torch.tensor(values,         dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        policy  = torch.zeros(POLICY_SIZE, dtype=torch.float32)
        indices = self.policy_indices[idx]
        probs   = self.policy_probs[idx]
        mask    = indices >= 0
        if mask.any():
            policy.scatter_(0, indices[mask], probs[mask])
        return self.states[idx], policy, self.values[idx]


# =========================
# Supervised Dataset
# =========================

class SupervisedDataset(Dataset):
    """Wraps the existing Lichess NPZ dataset."""

    def __init__(self, npz_path):
        data                 = np.load(npz_path)
        self.states          = torch.tensor(data["states"],         dtype=torch.float32)
        self.policy_indices  = torch.tensor(data["policy_indices"], dtype=torch.long)
        self.policy_probs    = torch.tensor(data["policy_probs"],   dtype=torch.float32)
        self.values          = torch.tensor(data["values"],         dtype=torch.float32)
        print(f"Supervised dataset loaded: {len(self.states):,} positions")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        policy  = torch.zeros(POLICY_SIZE, dtype=torch.float32)
        indices = self.policy_indices[idx]
        probs   = self.policy_probs[idx]
        mask    = indices >= 0
        if mask.any():
            policy.scatter_(0, indices[mask], probs[mask])
        return self.states[idx], policy, self.values[idx]


# =========================
# Loss Functions
# =========================

def policy_kl_loss(logits, soft_targets):
    """KL divergence loss for soft policy targets."""
    log_probs = F.log_softmax(logits, dim=1)
    loss      = -(soft_targets * log_probs).sum(dim=1).mean()
    return loss


# =========================
# Training Function
# =========================

def train_iteration(model, replay_buffer, iteration, device):
    """
    Train the model for one self-play iteration.

    Mixes self-play data from replay buffer with supervised Lichess data
    according to the mix schedule. Returns training metrics.

    Args:
        model:          ChessNet2 instance (will be modified in place)
        replay_buffer:  ReplayBuffer with self-play positions
        iteration:      current iteration number (affects mix schedule)
        device:         torch device

    Returns:
        dict with avg_policy_loss, avg_value_loss, avg_total_loss
    """

    supervised_frac = get_supervised_fraction(iteration)
    print(f"\nIteration {iteration} training:")
    print(f"  Mix: {supervised_frac*100:.0f}% supervised + {(1-supervised_frac)*100:.0f}% self-play")
    print(f"  Replay buffer size: {len(replay_buffer):,} positions")

    # ---- Build self-play dataset ----
    # Sample all positions from buffer (training uses full buffer each iteration)
    buffer_size = len(replay_buffer)
    if buffer_size == 0:
        print("  Warning: empty replay buffer — skipping self-play data")
        supervised_frac = 1.0

    # ---- Load supervised dataset ----
    supervised_dataset = None
    if supervised_frac > 0 and os.path.exists(SUPERVISED_NPZ):
        supervised_dataset = SupervisedDataset(SUPERVISED_NPZ)

    # ---- Create mixed dataloader ----
    # We interleave batches from both datasets according to the mix schedule
    is_linux    = os.name != 'nt' and hasattr(os, 'uname') and os.uname().sysname == 'Linux'
    num_workers = 2 if is_linux else 0

    # Self-play loader
    sp_loader = None
    if buffer_size > 0:
        sp_states, sp_indices, sp_probs, sp_values = replay_buffer.sample(buffer_size)
        sp_dataset = SelfPlayDataset(sp_states, sp_indices, sp_probs, sp_values)
        sp_loader  = DataLoader(
            sp_dataset, batch_size=BATCH_SIZE,
            shuffle=True, num_workers=num_workers, pin_memory=True
        )

    # Supervised loader
    sv_loader = None
    if supervised_dataset is not None:
        sv_loader = DataLoader(
            supervised_dataset, batch_size=BATCH_SIZE,
            shuffle=True, num_workers=num_workers, pin_memory=True
        )

    # ---- Optimizer ----
    optimizer    = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    value_loss_fn = nn.MSELoss()

    # ---- Training loop ----
    model.train()

    total_policy = total_value = total_loss = 0.0
    total_batches = 0

    for epoch in range(EPOCHS):
        epoch_policy = epoch_value = epoch_total = 0.0
        epoch_batches = 0

        # Create iterators for this epoch
        sp_iter = iter(sp_loader) if sp_loader else None
        sv_iter = iter(sv_loader) if sv_loader else None

        # Determine number of batches this epoch
        n_sp_batches = len(sp_loader) if sp_loader else 0
        n_sv_batches = len(sv_loader) if sv_loader else 0

        # Interleave batches according to mix schedule
        # For each self-play batch, also process supervised_frac/(1-supervised_frac) supervised batches
        if supervised_frac >= 1.0:
            # Pure supervised
            batches = [("sv", None) for _ in range(n_sv_batches)]
        elif supervised_frac <= 0.0:
            # Pure self-play
            batches = [("sp", None) for _ in range(n_sp_batches)]
        else:
            # Mixed — interleave proportionally
            total_batches_epoch = max(n_sp_batches, n_sv_batches)
            batches = []
            for i in range(total_batches_epoch):
                if np.random.random() < supervised_frac and sv_iter:
                    batches.append("sv")
                elif sp_iter:
                    batches.append("sp")
                elif sv_iter:
                    batches.append("sv")

        for source in batches:
            try:
                if source == "sv" and sv_iter:
                    states_b, policies_b, values_b = next(sv_iter)
                elif source == "sp" and sp_iter:
                    states_b, policies_b, values_b = next(sp_iter)
                else:
                    continue
            except StopIteration:
                continue

            states_b   = states_b.to(device)
            policies_b = policies_b.to(device)
            values_b   = values_b.to(device)

            optimizer.zero_grad()

            pred_policy, pred_value = model(states_b)

            p_loss = policy_kl_loss(pred_policy, policies_b)
            v_loss = value_loss_fn(pred_value.view(-1), values_b)
            loss   = p_loss + v_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_policy  += p_loss.item()
            epoch_value   += v_loss.item()
            epoch_total   += loss.item()
            epoch_batches += 1

        if epoch_batches > 0:
            print(
                f"  Epoch {epoch+1}/{EPOCHS} | "
                f"policy={epoch_policy/epoch_batches:.4f}  "
                f"value={epoch_value/epoch_batches:.4f}  "
                f"total={epoch_total/epoch_batches:.4f}"
            )
            total_policy  += epoch_policy  / epoch_batches
            total_value   += epoch_value   / epoch_batches
            total_loss    += epoch_total   / epoch_batches
            total_batches += 1

    metrics = {
        "avg_policy_loss": total_policy / max(total_batches, 1),
        "avg_value_loss":  total_value  / max(total_batches, 1),
        "avg_total_loss":  total_loss   / max(total_batches, 1),
    }

    print(f"\n  Final avg: policy={metrics['avg_policy_loss']:.4f}  "
          f"value={metrics['avg_value_loss']:.4f}  "
          f"total={metrics['avg_total_loss']:.4f}")

    return metrics