import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):

    def __init__(self, channels, dropout=0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))

        x += residual
        x = F.relu(x)

        return x


class ChessNet2(nn.Module):

    def __init__(self):
        super().__init__()

        channels = 128

        self.conv = nn.Conv2d(18, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels, dropout=0.1) for _ in range(10)]
        )

        self.policy_conv = nn.Conv2d(channels, 73, 1)

        self.value_conv = nn.Conv2d(channels, 8, 1)
        self.value_fc1 = nn.Linear(8 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):

        x = F.relu(self.bn(self.conv(x)))

        x = self.res_blocks(x)

        p = self.policy_conv(x)
        policy = torch.flatten(p, 1)

        v = F.relu(self.value_conv(x))
        v = torch.flatten(v, 1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value

class ChessNet1(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(18, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)

        self.policy_head = nn.Linear(64 * 8 * 8, 4672)

        self.value_head = nn.Linear(64 * 8 * 8, 1)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.flatten(x, 1)

        policy = self.policy_head(x)

        value = torch.tanh(self.value_head(x))

        return policy, value
