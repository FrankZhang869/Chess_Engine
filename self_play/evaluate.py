"""
evaluate.py
===========
Pits the newly trained model against the previous best model
to decide whether to promote the new model.

Promotion threshold: new model must win > 55% of games.
This threshold prevents promoting a model that got lucky
in a small sample — 55% over 40 games is statistically meaningful.

Games are played with lower simulations (100) to keep evaluation fast.
Color is alternated so neither model gets an unfair advantage.
"""

import torch
import chess
import numpy as np

from model import ChessNet2
from mcts import mcts_search


# =========================
# Configuration
# =========================

EVAL_GAMES       = 40     # total games in evaluation match
EVAL_SIMULATIONS = 100    # sims per move during evaluation (lower = faster)
PROMOTION_THRESHOLD = 0.55  # new model must win > 55% to be promoted


# =========================
# Play One Game
# =========================

def play_game(white_model, black_model, device, simulations):
    """
    Play one game between two models.

    Returns:
        1  if white wins
        -1 if black wins
        0  for draw
    """
    board = chess.Board()

    while not board.is_game_over():

        if board.fullmove_number > 150:
            # Prevent infinite games
            return 0

        if board.turn == chess.WHITE:
            move = mcts_search(board, white_model, device,
                               simulations=simulations, temperature=0)
        else:
            move = mcts_search(board, black_model, device,
                               simulations=simulations, temperature=0)

        if move is None:
            break

        board.push(move)

    result = board.result()
    if result == "1-0":  return 1
    if result == "0-1":  return -1
    return 0


# =========================
# Run Evaluation Match
# =========================

def evaluate(new_model_path, best_model_path, device):
    """
    Run a match between new and best model.
    Alternates colors every game for fairness.

    Args:
        new_model_path:  path to newly trained model weights
        best_model_path: path to current best model weights
        device:          torch device

    Returns:
        (promoted, win_rate) tuple
            promoted: True if new model should replace best model
            win_rate: fraction of games won by new model (draws = 0.5)
    """
    print(f"\nEvaluating: {new_model_path} vs {best_model_path}")
    print(f"Games: {EVAL_GAMES}  Simulations: {EVAL_SIMULATIONS}")
    print(f"Promotion threshold: {PROMOTION_THRESHOLD*100:.0f}%\n")

    # Load models
    new_model  = ChessNet2().to(device)
    best_model = ChessNet2().to(device)

    new_model.load_state_dict(torch.load(new_model_path,  map_location=device))
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))

    new_model.eval()
    best_model.eval()

    wins = draws = losses = 0

    for game_num in range(EVAL_GAMES):
        # Alternate colors every game
        new_is_white = (game_num % 2 == 0)

        if new_is_white:
            result = play_game(new_model, best_model, device, EVAL_SIMULATIONS)
            # result is from white's perspective
            if result == 1:   wins   += 1
            elif result == 0: draws  += 1
            else:             losses += 1
            outcome = "Win" if result == 1 else "Draw" if result == 0 else "Loss"
        else:
            result = play_game(best_model, new_model, device, EVAL_SIMULATIONS)
            # result is from white's perspective — new model is black
            if result == -1:  wins   += 1
            elif result == 0: draws  += 1
            else:             losses += 1
            outcome = "Win" if result == -1 else "Draw" if result == 0 else "Loss"

        print(f"  Game {game_num+1:02d}/{EVAL_GAMES} | "
              f"New={'White' if new_is_white else 'Black'} | "
              f"{outcome} | Score: +{wins} ={draws} -{losses}")

    win_rate = (wins + 0.5 * draws) / EVAL_GAMES
    promoted = win_rate > PROMOTION_THRESHOLD

    print(f"\nFinal score: +{wins} ={draws} -{losses}")
    print(f"Win rate: {win_rate*100:.1f}%  (threshold: {PROMOTION_THRESHOLD*100:.0f}%)")
    print(f"Decision: {'PROMOTED ✓' if promoted else 'REJECTED ✗'}")

    return promoted, win_rate