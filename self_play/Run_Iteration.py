"""
run_iteration.py
================
Orchestrates one complete self-play iteration:
    1. Load current best model
    2. Generate self-play games (batched MCTS)
    3. Add games to replay buffer
    4. Train new model on replay buffer + supervised data
    5. Evaluate new model against current best
    6. Promote if win rate > 55%
    7. Save checkpoint and buffer

Run this script once per iteration on Colab:
    python run_iteration.py --iteration 1

Each iteration takes roughly 3-4 hours on a Colab T4 GPU:
    Game generation:  ~2.5 hours  (200 games × ~45s/game)
    Training:         ~25 minutes
    Evaluation:       ~15 minutes

Progress is printed throughout so you can monitor in Colab.
"""

import argparse
import json
import shutil
import torch
from pathlib import Path

from model import ChessNet2
from self_play.self_play import BatchedSelfPlay
from self_play.replay_buffer import ReplayBuffer
from self_play.train_self_play import train_iteration
from self_play.evaluate import evaluate


# =========================
# Configuration
# =========================

GAMES_PER_ITERATION = 200
BEST_MODEL_PATH     = "best_model.pt"
ITERATION_LOG       = "iteration_log.json"
BUFFER_PATH         = "replay_buffer.pkl"


# =========================
# Main
# =========================

def run_iteration(iteration):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"\n{'='*60}")
    print(f"SELF-PLAY ITERATION {iteration}")
    print(f"{'='*60}\n")

    # ---- Step 1: Load current best model ----
    print(f"Step 1: Loading current best model from {BEST_MODEL_PATH}")
    model = ChessNet2().to(device)

    if Path(BEST_MODEL_PATH).exists():
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        print(f"  Loaded existing best model.")
    else:
        print(f"  No existing model found — starting from supervised checkpoint.")
        print(f"  Please ensure best_model.pt exists before running self-play.")
        return

    model.eval()

    # ---- Step 2: Load replay buffer ----
    print(f"\nStep 2: Loading replay buffer")
    replay_buffer = ReplayBuffer()
    replay_buffer.load(BUFFER_PATH)

    # ---- Step 3: Generate self-play games ----
    print(f"\nStep 3: Generating {GAMES_PER_ITERATION} self-play games")
    generator = BatchedSelfPlay(model, device)
    game_results = generator.run_games(GAMES_PER_ITERATION, replay_buffer)

    # Save buffer after generation
    replay_buffer.save(BUFFER_PATH)
    print(f"  Replay buffer now contains {len(replay_buffer):,} positions")

    # ---- Step 4: Train new model ----
    print(f"\nStep 4: Training new model")

    # Start from current best model weights
    new_model = ChessNet2().to(device)
    new_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    metrics = train_iteration(new_model, replay_buffer, iteration, device)

    # Save new model for evaluation
    new_model_path = f"model_iter_{iteration}.pt"
    torch.save(new_model.state_dict(), new_model_path)
    print(f"  New model saved to {new_model_path}")

    # ---- Step 5: Evaluate new model ----
    print(f"\nStep 5: Evaluating new model vs current best")
    promoted, win_rate = evaluate(new_model_path, BEST_MODEL_PATH, device)

    # ---- Step 6: Promote if better ----
    if promoted:
        print(f"\nPromoting new model to best_model.pt")
        shutil.copy(new_model_path, BEST_MODEL_PATH)
        print(f"  best_model.pt updated.")
    else:
        print(f"\nKeeping existing best_model.pt (new model rejected)")

    # ---- Step 7: Log results ----
    log = []
    if Path(ITERATION_LOG).exists():
        with open(ITERATION_LOG, "r") as f:
            log = json.load(f)

    log.append({
        "iteration":       iteration,
        "games_generated": GAMES_PER_ITERATION,
        "white_wins":      game_results["white_wins"],
        "black_wins":      game_results["black_wins"],
        "draws":           game_results["draws"],
        "avg_game_length": game_results["total_moves"] / GAMES_PER_ITERATION,
        "train_loss":      metrics["avg_total_loss"],
        "policy_loss":     metrics["avg_policy_loss"],
        "value_loss":      metrics["avg_value_loss"],
        "eval_win_rate":   win_rate,
        "promoted":        promoted,
        "buffer_size":     len(replay_buffer),
    })

    with open(ITERATION_LOG, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"ITERATION {iteration} COMPLETE")
    print(f"  Win rate vs previous best: {win_rate*100:.1f}%")
    print(f"  Promoted: {promoted}")
    print(f"  Training loss: {metrics['avg_total_loss']:.4f}")
    print(f"  Buffer size: {len(replay_buffer):,} positions")
    print(f"  Log saved to {ITERATION_LOG}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iteration", type=int, required=True,
        help="Iteration number (1, 2, 3, ...)"
    )
    args = parser.parse_args()
    run_iteration(args.iteration)