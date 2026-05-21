import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Residual Block
# =========================

class ResidualBlock(nn.Module):
    """
    A single residual block — the core building unit of the network tower.

    Structure:
        input
          → Conv2d (3x3, same spatial size)
          → BatchNorm
          → ReLU
          → Dropout2d          ← regularisation to reduce overfitting
          → Conv2d (3x3, same spatial size)
          → BatchNorm
          → Add input (skip connection)
          → ReLU
          → output

    The skip connection (input added back before the final ReLU) is what makes
    this "residual". It allows gradients to flow directly back through the network
    during training without vanishing, which is why deep residual networks train
    much better than plain deep networks.

    Dropout2d randomly zeroes entire feature map channels during training.
    This forces the network to learn redundant representations and prevents
    it from memorising specific training positions. p=0.1 zeroes 10% of
    channels per forward pass — aggressive enough to regularise without
    hurting learning speed.
    Dropout is automatically disabled during model.eval() — so inference
    and MCTS are not affected.

    bias=False on Conv2d because BatchNorm already handles the bias term —
    having both wastes parameters.
    """

    def __init__(self, channels, dropout=0.1):
        super().__init__()

        self.conv1   = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(p=dropout)
        self.conv2   = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2     = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x                                    # save input for skip connection

        x = F.relu(self.bn1(self.conv1(x)))             # first conv + norm + activation
        x = self.dropout(x)                             # dropout after first activation
        x = self.bn2(self.conv2(x))                     # second conv + norm (no ReLU yet)

        x += residual                                   # skip connection
        x = F.relu(x)                                   # activation after adding skip

        return x


# =========================
# Main Network
# =========================

class ChessNet2(nn.Module):
    """
    AlphaZero-style chess network with:
      - Input stem:      18-plane board representation → 128 feature channels
      - Residual tower:  10 residual blocks at 128 channels (with dropout=0.1)
      - Policy head:     outputs 4672 move logits (8x8x73 AlphaZero encoding)
      - Value head:      outputs a scalar in (-1, 1) representing position evaluation

    Size: ~5M parameters — matches LC0's smallest competitive network configuration
    (10 blocks × 128 filters), as documented at lczero.org/dev/backend/nn/

    Input shape:   (batch, 18, 8, 8)
    Policy output: (batch, 4672)  — raw logits, NOT softmaxed
    Value output:  (batch, 1)     — tanh-squashed scalar
    """

    def __init__(self):
        super().__init__()

        channels = 128      # upgraded from 96 → matches LC0 smallest network width

        # -------------------------
        # Input Stem
        # -------------------------
        # Takes the 18-plane board tensor and projects it into 128 feature channels.
        # 3x3 conv with padding=1 preserves the 8x8 spatial dimensions.
        # bias=False because BatchNorm handles bias.
        self.conv = nn.Conv2d(18, channels, 3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(channels)

        # -------------------------
        # Residual Tower
        # -------------------------
        # 10 stacked residual blocks, each maintaining 128 channels and 8x8 spatial size.
        # Upgraded from 8×96 to 10×128 — doubles parameter count from ~2.5M to ~5M.
        # Each block includes dropout=0.1 to reduce overfitting on the 500k dataset.
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels, dropout=0.1) for _ in range(10)])

        # -------------------------
        # Policy Head
        # -------------------------
        # Predicts a probability distribution over all legal moves.
        #
        # Architecture:
        #   128ch feature map → 1x1 conv → 73ch map → flatten → 4672 logits
        #
        # The 73 output channels correspond to AlphaZero's move encoding:
        #   - 56 channels: sliding moves (8 directions × 7 distances)
        #   - 8 channels:  knight moves
        #   - 9 channels:  underpromotions (3 piece types × 3 directions)
        # The 8x8 spatial dimensions represent the from-square.
        # Flattening 73 × 8 × 8 = 4672 covers the full action space.
        #
        # Output is raw logits — softmax is applied in the loss function,
        # not here, so that numerical stability is handled correctly.
        self.policy_conv = nn.Conv2d(channels, 73, 1)

        # -------------------------
        # Value Head
        # -------------------------
        # Predicts how good the position is for the current player: -1 = losing, +1 = winning.
        #
        # Architecture:
        #   128ch feature map
        #   → 1x1 conv (128ch → 8ch)    compress channels
        #   → flatten (8 × 8 × 8 = 512)
        #   → Linear (512 → 128)         learn evaluation features
        #   → ReLU
        #   → Linear (128 → 1)           single scalar output
        #   → tanh                       squash to (-1, 1)
        #
        # Using 8 output channels in the conv gives the head more expressive
        # power — it can extract 8 different spatial summaries of the position
        # before collapsing to a scalar.
        self.value_conv = nn.Conv2d(channels, 8, 1)
        self.value_fc1  = nn.Linear(8 * 8 * 8, 128)
        self.value_fc2  = nn.Linear(128, 1)

    def forward(self, x):

        # --- Stem ---
        x = F.relu(self.bn(self.conv(x)))   # (B, 18, 8, 8)  → (B, 128, 8, 8)

        # --- Residual Tower ---
        x = self.res_blocks(x)              # (B, 128, 8, 8) → (B, 128, 8, 8)

        # --- Policy Head ---
        p      = self.policy_conv(x)        # (B, 128, 8, 8) → (B, 73, 8, 8)
        policy = torch.flatten(p, 1)        # (B, 73, 8, 8)  → (B, 4672)

        # --- Value Head ---
        v     = F.relu(self.value_conv(x))     # (B, 128, 8, 8) → (B, 8, 8, 8)
        v     = torch.flatten(v, 1)            # (B, 8, 8, 8)   → (B, 512)
        v     = F.relu(self.value_fc1(v))      # (B, 512)       → (B, 128)
        value = torch.tanh(self.value_fc2(v))  # (B, 128)       → (B, 1)

        return policy, value

class ChessNet1(nn.Module):

    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(18, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)

        # POLICY HEAD (matches saved model)
        self.policy_head = nn.Linear(64 * 8 * 8, 4672)

        # VALUE HEAD (matches saved model)
        self.value_head = nn.Linear(64 * 8 * 8, 1)

    def forward(self, x):

        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.flatten(x, 1)

        # Policy output
        policy = self.policy_head(x)

        # Value output
        value = torch.tanh(self.value_head(x))

        return policy, value
    
