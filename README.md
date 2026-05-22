# AlphaZero-Inspired Chess Engine

A deep reinforcement learning chess engine built from scratch using PyTorch, Monte Carlo Tree Search (MCTS), supervised learning from high-level human games, and iterative self-play training.

The project draws inspiration from DeepMind's AlphaZero architecture while being designed and trained entirely on consumer hardware. The goal is to investigate how far modern neural-network-based chess systems can be pushed under realistic computational constraints and to study the effectiveness of self-play reinforcement learning at small scale.

---

## Features

- AlphaZero-style residual neural network
- Dual-head architecture:
  - Policy head predicts move probabilities
  - Value head evaluates board positions
- Monte Carlo Tree Search (PUCT)
- Batched self-play generation
- Replay buffer with sliding-window sampling
- Supervised pretraining on human games
- Reinforcement learning through iterative self-play
- Automatic model promotion system
- GPU-accelerated training and inference
- Full chess move encoding (4672-action AlphaZero representation)

---

## Architecture

### Neural Network

The engine uses a residual convolutional neural network implemented in PyTorch.

**Input Representation**

- 18-channel board encoding
- Piece planes
- Castling rights
- En passant information
- Side-to-move normalization through board mirroring

**Network Structure**

- Initial convolutional stem
- 10 residual blocks
- 128 feature channels
- Policy head (4672 move logits)
- Value head (scalar evaluation in [-1, 1])

Approximately 5 million trainable parameters.

---

### Monte Carlo Tree Search

Move selection is performed using Monte Carlo Tree Search with:

- PUCT exploration formula
- Neural-network priors
- Value-guided rollouts
- Root Dirichlet noise during self-play
- Tree reuse between moves
- Batched neural evaluations for efficiency

The search replaces traditional handcrafted evaluation functions and allows the engine to combine learned positional understanding with lookahead search.

---

## Dataset

### Human Game Dataset

The initial supervised training phase uses positions extracted from public Lichess games.

Selection criteria:

- Player rating approximately 1800+ Elo
- High-quality human play
- Diverse openings and middlegame structures

This provides the policy network with a strong prior before reinforcement learning begins.

---

### Tactical Puzzle Dataset

To improve tactical awareness, training data was augmented with chess puzzles emphasizing:

- Checkmate patterns
- Forced tactical sequences
- Mating attacks
- Tactical motifs

The objective was to expose the model to tactical concepts that appear infrequently in ordinary game data.

---

### Stockfish Evaluation Targets

In addition to move-selection supervision, positions were analyzed using Stockfish to generate evaluation targets.

These evaluations are used to train the value head, allowing the network to learn positional assessment beyond simple game outcomes.

This creates a dense learning signal compared to relying solely on win/loss labels.

---

## Training Pipeline

### Stage 1 — Supervised Learning

The network is first trained on:

- Human move choices
- Stockfish evaluations
- Puzzle positions

Losses:

- Policy loss (soft target distribution)
- Value loss (position evaluation)

This stage establishes baseline chess knowledge before self-play begins.

---

### Stage 2 — Self-Play Reinforcement Learning

After supervised pretraining, the engine enters iterative self-play training.

For each iteration:

1. Current best model generates self-play games
2. Search visit distributions become policy targets
3. Positions are stored in a replay buffer
4. A new network is trained on:
   - Historical supervised data
   - Recent self-play positions
5. The new network is evaluated against the current best model
6. Promotion occurs only if performance improves

This process follows the general training philosophy introduced by AlphaZero while operating under significantly smaller computational budgets.

---

## Model Promotion System

Each candidate model must defeat the current best model in head-to-head evaluation matches before being promoted.

Evaluation procedure:

- Fixed number of games
- Alternating colors
- Equal search budget
- Promotion threshold above 50% score

This prevents weaker updates from replacing stronger networks.

---

## Reinforcement Learning Infrastructure

Implemented components:

- Replay buffer
- Batched self-play generation
- Search-policy extraction
- Automated checkpointing
- Candidate model evaluation
- Promotion gating
- Iteration logging

The entire pipeline can run autonomously and continuously generate new training data.

---

## Research Motivation

A central objective of this project is exploring the practical limitations of AlphaZero-style learning under consumer-level compute constraints.

Questions investigated include:

- How effective is self-play when compute resources are limited?
- Can reinforcement learning improve upon supervised chess knowledge at small scale?
- How much tactical strength emerges from self-play alone?
- What role does search play relative to learned evaluation?

The project serves both as an engineering exercise and as an experimental platform for studying modern reinforcement learning systems.

---

## Repository History

This repository was created after substantial development had already occurred locally.

As a result, the commit history does not fully reflect the chronological development of the project.

Progress can instead be tracked through:

- Model checkpoints
- Training logs
- Dataset revisions
- Self-play iteration history
- Evaluation results

These artifacts document the evolution of the engine across multiple training generations and architectural revisions.

---

## Technologies

- Python
- PyTorch
- NumPy
- python-chess
- CUDA
- Monte Carlo Tree Search
- Deep Residual Networks
- Reinforcement Learning
- Self-Play Training

---

## Future Work

Potential future directions include:

- Alpha-Beta search comparisons
- Enhanced tactical training methods
- Value-head calibration studies
- Larger-scale self-play experiments
- Search optimization
- Alternative network architectures
- Hybrid neural/search systems

---

## Disclaimer

This project is an independent educational and research effort inspired by the ideas introduced in AlphaZero. It is not affiliated with DeepMind, Google, Lichess, Stockfish, or Leela Chess Zero.