"""import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from model import ChessNet2


# =========================
# Load dataset
# =========================

data = np.load("dataset.npz")

states = data["states"]
moves = data["moves"]
values = data["values"]


# =========================
# Dataset class
# =========================

class ChessDataset(Dataset):

    def __init__(self, states, moves, values):

        self.states = torch.tensor(states, dtype=torch.float32)
        self.moves = torch.tensor(moves, dtype=torch.long)
        self.values = torch.tensor(values, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.moves[idx], self.values[idx]


dataset = ChessDataset(states, moves, values)


# =========================
# DataLoader
# =========================

dataloader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)


# =========================
# Device
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================
# Model
# =========================

model = ChessNet2().to(device)

param_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {param_count:,}")


# =========================
# Loss functions
# =========================

policy_loss_fn = nn.CrossEntropyLoss()
value_loss_fn = nn.MSELoss()


# =========================
# Optimizer
# =========================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005
)


# =========================
# Training parameters
# =========================

epochs = 20
best_loss = float("inf")


# =========================
# Training loop
# =========================

for epoch in range(epochs):

    model.train()

    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0

    for states, moves, values in dataloader:

        states = states.to(device)
        moves = moves.to(device)
        values = values.to(device)

        optimizer.zero_grad()

        pred_policy, pred_value = model(states)

        policy_loss = policy_loss_fn(pred_policy, moves)

        value_loss = value_loss_fn(
            pred_value.view(-1),
            values
        )

        loss = policy_loss + 0.5 * value_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_policy = total_policy_loss / len(dataloader)
    avg_value = total_value_loss / len(dataloader)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {avg_loss:.4f} | "
        f"Policy: {avg_policy:.4f} | "
        f"Value: {avg_value:.4f}"
    )

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_model.pt")
        print("Saved new best model.")


torch.save(model.state_dict(), "chess_model.pt")

print("Training complete.")
print("Best model saved as best_model.pt")
print("Final model saved as chess_model.pt")"""

# new train:
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

from model import ChessNet2


# =========================
# Configuration
# =========================

POLICY_SIZE = 64 * 73   # 4672 — total number of possible moves in AlphaZero encoding
BATCH_SIZE  = 256        # positions per gradient update
                         # 512 is faster but may cause out-of-memory on Colab free tier
                         # drop to 128 if you see CUDA OOM errors
EPOCHS      = 30         # maximum training epochs (early stopping may end it sooner)
LR          = 1e-3       # initial learning rate — decays to 1e-5 via cosine schedule
WEIGHT_DECAY = 3e-4      # L2 regularization — penalizes large weights to reduce overfitting
PATIENCE    = 10         # early stopping patience — how many epochs of no val improvement
                         # before stopping. 10 is safer than 5 with cosine LR schedule
                         # because loss can plateau mid-schedule before improving again


# =========================
# Dataset Class
# =========================

class ChessDataset(torch.utils.data.Dataset):
    """
    Loads the preprocessed .npz dataset and serves (state, policy, value) tuples.

    Data stored in sparse format to save disk space:
      states:          (N, 18, 8, 8) int8   — board representation
      policy_indices:  (N, 5)        int16  — indices of top-5 moves in the 4672 action space
      policy_probs:    (N, 5)        float32 — probabilities for those top-5 moves
      values:          (N,)          float32 — tanh-scaled Stockfish evaluation

    In __getitem__, the sparse policy is reconstructed into a dense 4672-float vector
    using scatter_. This reconstruction is very fast (microseconds) and means we
    store ~6MB of policy data instead of ~5GB.

    Padding: unused slots in policy_indices are stored as -1 with prob 0.0.
    The mask (indices >= 0) filters these out before scattering.
    """

    def __init__(self, states, policy_indices, policy_probs, values):
        self.states         = torch.tensor(states,         dtype=torch.float32)
        self.policy_indices = torch.tensor(policy_indices, dtype=torch.long)
        self.policy_probs   = torch.tensor(policy_probs,   dtype=torch.float32)
        self.values         = torch.tensor(values,         dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # Reconstruct sparse → dense policy vector
        policy  = torch.zeros(POLICY_SIZE, dtype=torch.float32)
        indices = self.policy_indices[idx]      # (5,) — move indices
        probs   = self.policy_probs[idx]        # (5,) — probabilities

        mask = indices >= 0                     # filter out padding slots (-1)
        policy.scatter_(0, indices[mask], probs[mask])

        return self.states[idx], policy, self.values[idx]


# =========================
# Loss Function: KL Divergence
# =========================

def policy_kl_loss(logits, soft_targets):
    """
    KL divergence loss between the network's predicted policy and the soft target
    distribution from Stockfish MultiPV analysis.

    Why KL divergence instead of cross entropy:
      Cross entropy uses one-hot targets (one correct move, everything else = 0).
      This gives the network zero signal about whether the 2nd-best move is
      almost as good or completely wrong.

      KL divergence uses a full probability distribution as the target
      (e.g. move A=45%, move B=30%, move C=20%), so the network learns the
      full ranking of moves, not just the best one.

    Math:
      KL(target || predicted) = sum(target * log(target/predicted))
                              = sum(target * log(target)) - sum(target * log(predicted))

      The first term is constant w.r.t. model parameters, so minimising KL
      is equivalent to minimising -sum(target * log_probs), which is what
      we compute here.

    Args:
      logits:       (B, 4672) raw network output — NOT softmaxed
      soft_targets: (B, 4672) target probability distribution, sums to 1 per row

    Returns:
      Scalar loss value
    """
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(soft_targets * log_probs).sum(dim=1).mean()
    return loss


# =========================
# Load Dataset
# =========================

print("Loading dataset ...")
data           = np.load("dataset.npz")
states         = data["states"]
policy_indices = data["policy_indices"]
policy_probs   = data["policy_probs"]
values         = data["values"]
print(f"Loaded {len(states):,} positions.")

dataset = ChessDataset(states, policy_indices, policy_probs, values)


# =========================
# Train / Validation Split
# =========================
#
# 90% of data used for training, 10% held out for validation.
# Validation set is never used for gradient updates — it measures how well
# the model generalises to positions it hasn't trained on.
#
# If train loss keeps decreasing but val loss starts increasing, the model
# is overfitting (memorising training positions rather than learning chess).
# Early stopping uses val loss to catch this automatically.

val_size   = int(0.10 * len(dataset))
train_size = len(dataset) - val_size

train_set, val_set = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # fixed seed = reproducible split
)
print(f"Train: {train_size:,}   Val: {val_size:,}")


# =========================
# DataLoader
# =========================
#
# num_workers controls how many CPU processes prefetch data in parallel.
# PyTorch multiprocessing on macOS frequently causes crashes/hangs,
# so we use 0 workers on Mac (data loaded in main process) and 2 on Linux (Colab).
# pin_memory=True speeds up CPU→GPU transfers on Colab by using pinned memory.

is_linux    = os.name != 'nt' and hasattr(os, 'uname') and os.uname().sysname == 'Linux'
num_workers = 2 if is_linux else 0
print(f"DataLoader workers: {num_workers} ({'Colab/Linux' if is_linux else 'Mac'})")

train_loader = DataLoader(
    train_set,
    batch_size  = BATCH_SIZE,
    shuffle     = True,           # shuffle so model doesn't see positions from same game consecutively
    num_workers = num_workers,
    pin_memory  = True
)

val_loader = DataLoader(
    val_set,
    batch_size  = BATCH_SIZE,
    shuffle     = False,          # no need to shuffle validation
    num_workers = num_workers,
    pin_memory  = True
)


# =========================
# Device
# =========================
#
# Automatically uses GPU if available (Colab), otherwise CPU (Mac).
# Training on CPU is ~10-50x slower — always use Colab for training.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =========================
# Model
# =========================

model = ChessNet2().to(device)
param_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {param_count:,}")


# =========================
# Optimizer
# =========================
#
# Adam: adaptive learning rate optimizer — adjusts lr per parameter based on
# gradient history. Works well for most deep learning tasks.
#
# weight_decay: L2 regularization — adds a small penalty for large weights,
# which discourages overfitting. 1e-4 is a standard value.

optimizer = torch.optim.Adam(
    model.parameters(),
    lr           = LR,
    weight_decay = WEIGHT_DECAY
)


# =========================
# Learning Rate Scheduler
# =========================
#
# CosineAnnealingLR: smoothly decays learning rate from LR (1e-3) to eta_min (1e-5)
# following a cosine curve over T_max epochs.
#
# Why cosine decay:
#   - High LR early in training = fast progress through the loss landscape
#   - Low LR late in training = fine-grained convergence to a good minimum
#   - Cosine shape avoids sudden drops which can destabilise training
#
# The plateau behaviour: lr sometimes stays flat for several epochs mid-schedule
# before improving. This is why patience=10 (not 5) — so early stopping
# doesn't fire during a natural LR plateau.

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max   = EPOCHS,
    eta_min = 1e-5
)

value_loss_fn = nn.MSELoss()
# MSELoss for value head: penalises squared distance between predicted
# and target evaluation. Standard choice for regression to a scalar.


# =========================
# Training Loop
# =========================

best_val_loss  = float("inf")
patience_count = 0

for epoch in range(1, EPOCHS + 1):

    # ---- Training Phase ----
    model.train()   # enables dropout/batchnorm training behaviour

    t_policy = t_value = t_total = 0.0
    grad_norms = []

    for batch_idx, (states_b, policies_b, values_b) in enumerate(train_loader):

        # Move tensors to GPU (no-op if already on CPU)
        states_b   = states_b.to(device)
        policies_b = policies_b.to(device)
        values_b   = values_b.to(device)

        optimizer.zero_grad()   # clear gradients from previous batch

        # Forward pass
        pred_policy, pred_value = model(states_b)

        # Compute losses
        p_loss = policy_kl_loss(pred_policy, policies_b)
        v_loss = value_loss_fn(pred_value.view(-1), values_b)
        loss   = p_loss + v_loss    # equal weighting of policy and value loss

        # Backward pass
        loss.backward()

        # Gradient clipping: prevents exploding gradients by scaling down
        # the gradient vector if its norm exceeds max_norm=1.0.
        # Returns the norm before clipping so we can monitor it.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norms.append(grad_norm.item())

        optimizer.step()    # update weights

        t_policy += p_loss.item()
        t_value  += v_loss.item()
        t_total  += loss.item()

    n = len(train_loader)
    train_policy = t_policy / n
    train_value  = t_value  / n
    train_total  = t_total  / n
    avg_grad_norm = sum(grad_norms) / len(grad_norms)

    # ---- Validation Phase ----
    model.eval()    # disables dropout/batchnorm training behaviour

    v_policy = v_value = v_total = 0.0

    with torch.no_grad():   # no gradients needed for validation = faster + less memory
        for states_b, policies_b, values_b in val_loader:
            states_b   = states_b.to(device)
            policies_b = policies_b.to(device)
            values_b   = values_b.to(device)

            pred_policy, pred_value = model(states_b)

            p_loss = policy_kl_loss(pred_policy, policies_b)
            v_loss = value_loss_fn(pred_value.view(-1), values_b)

            v_policy += p_loss.item()
            v_value  += v_loss.item()
            v_total  += (p_loss + v_loss).item()

    m = len(val_loader)
    val_policy = v_policy / m
    val_value  = v_value  / m
    val_total  = v_total  / m

    # Step LR scheduler after each epoch
    scheduler.step()
    lr = scheduler.get_last_lr()[0]

    # ---- Logging ----
    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train  policy={train_policy:.4f}  value={train_value:.4f}  total={train_total:.4f} | "
        f"Val    policy={val_policy:.4f}  value={val_value:.4f}  total={val_total:.4f} | "
        f"grad={avg_grad_norm:.3f}  lr={lr:.2e}"
    )

    # Gradient health check:
    # avg_grad_norm consistently > 5.0  → learning rate may be too high
    # avg_grad_norm consistently < 0.01 → learning has effectively stopped
    if avg_grad_norm > 5.0:
        print(f"  ⚠️  High gradient norm ({avg_grad_norm:.2f}) — consider reducing LR")
    if avg_grad_norm < 0.01:
        print(f"  ⚠️  Very low gradient norm ({avg_grad_norm:.4f}) — learning may have stalled")

    # ---- Checkpoint ----
    # Save model whenever validation loss improves.
    # Using val_total (not train_total) ensures we save the best-generalising
    # model, not the one that simply memorised training data best.
    if val_total < best_val_loss:
        best_val_loss  = val_total
        patience_count = 0
        torch.save(model.state_dict(), "best_model.pt")
        print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")
    else:
        patience_count += 1
        print(f"  No improvement ({patience_count}/{PATIENCE})")
        if patience_count >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

# ---- Final Save ----
torch.save(model.state_dict(), "chess_model.pt")
print(f"\nTraining complete.")
print(f"Best val loss : {best_val_loss:.4f}")
print(f"Best model    : best_model.pt")
print(f"Final model   : chess_model.pt")