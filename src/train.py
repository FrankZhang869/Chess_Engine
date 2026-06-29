import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

from model import ChessNet2


POLICY_SIZE = 64 * 73
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 3e-4
PATIENCE = 10


class ChessDataset(torch.utils.data.Dataset):

    def __init__(self, states, policy_indices, policy_probs, values):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.policy_indices = torch.tensor(policy_indices, dtype=torch.long)
        self.policy_probs = torch.tensor(policy_probs, dtype=torch.float32)
        self.values = torch.tensor(values, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        policy = torch.zeros(POLICY_SIZE, dtype=torch.float32)
        indices = self.policy_indices[idx]
        probs = self.policy_probs[idx]

        mask = indices >= 0
        policy.scatter_(0, indices[mask], probs[mask])

        return self.states[idx], policy, self.values[idx]


def policy_kl_loss(logits, soft_targets):
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(soft_targets * log_probs).sum(dim=1).mean()
    return loss


print("Loading dataset ...")
data = np.load("dataset.npz")
states = data["states"]
policy_indices = data["policy_indices"]
policy_probs = data["policy_probs"]
values = data["values"]
print(f"Loaded {len(states):,} positions.")

dataset = ChessDataset(states, policy_indices, policy_probs, values)


val_size = int(0.10 * len(dataset))
train_size = len(dataset) - val_size

train_set, val_set = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
print(f"Train: {train_size:,}   Val: {val_size:,}")


is_linux = os.name != 'nt' and hasattr(os, 'uname') and os.uname().sysname == 'Linux'
num_workers = 2 if is_linux else 0
print(f"DataLoader workers: {num_workers} ({'Colab/Linux' if is_linux else 'Mac'})")

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


model = ChessNet2().to(device)
param_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {param_count:,}")


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-5
)

value_loss_fn = nn.MSELoss()


best_val_loss = float("inf")
patience_count = 0

for epoch in range(1, EPOCHS + 1):

    model.train()

    t_policy = t_value = t_total = 0.0
    grad_norms = []

    for batch_idx, (states_b, policies_b, values_b) in enumerate(train_loader):

        states_b = states_b.to(device)
        policies_b = policies_b.to(device)
        values_b = values_b.to(device)

        optimizer.zero_grad()

        pred_policy, pred_value = model(states_b)

        p_loss = policy_kl_loss(pred_policy, policies_b)
        v_loss = value_loss_fn(pred_value.view(-1), values_b)
        loss = p_loss + v_loss

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norms.append(grad_norm.item())

        optimizer.step()

        t_policy += p_loss.item()
        t_value += v_loss.item()
        t_total += loss.item()

    n = len(train_loader)
    train_policy = t_policy / n
    train_value = t_value / n
    train_total = t_total / n
    avg_grad_norm = sum(grad_norms) / len(grad_norms)

    model.eval()

    v_policy = v_value = v_total = 0.0

    with torch.no_grad():
        for states_b, policies_b, values_b in val_loader:
            states_b = states_b.to(device)
            policies_b = policies_b.to(device)
            values_b = values_b.to(device)

            pred_policy, pred_value = model(states_b)

            p_loss = policy_kl_loss(pred_policy, policies_b)
            v_loss = value_loss_fn(pred_value.view(-1), values_b)

            v_policy += p_loss.item()
            v_value += v_loss.item()
            v_total += (p_loss + v_loss).item()

    m = len(val_loader)
    val_policy = v_policy / m
    val_value = v_value / m
    val_total = v_total / m

    scheduler.step()
    lr = scheduler.get_last_lr()[0]

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train  policy={train_policy:.4f}  value={train_value:.4f}  total={train_total:.4f} | "
        f"Val    policy={val_policy:.4f}  value={val_value:.4f}  total={val_total:.4f} | "
        f"grad={avg_grad_norm:.3f}  lr={lr:.2e}"
    )

    if avg_grad_norm > 5.0:
        print(f"High gradient norm ({avg_grad_norm:.2f})")
    if avg_grad_norm < 0.01:
        print(f"Very low gradient norm ({avg_grad_norm:.4f})")

    if val_total < best_val_loss:
        best_val_loss = val_total
        patience_count = 0
        torch.save(model.state_dict(), "best_model.pt")
        print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")
    else:
        patience_count += 1
        print(f"  No improvement ({patience_count}/{PATIENCE})")
        if patience_count >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

torch.save(model.state_dict(), "chess_model.pt")
print(f"\nTraining complete.")
print(f"Best val loss : {best_val_loss:.4f}")
print(f"Best model    : best_model.pt")
print(f"Final model   : chess_model.pt")
