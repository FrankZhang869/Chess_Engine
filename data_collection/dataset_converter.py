import pandas as pd
import chess
import numpy as np
from encoder import move_to_index, fen_to_tensor


# =========================
# Configuration
# =========================

MULTIPV      = 5          # number of top moves stored per position — must match generator
POLICY_SIZE  = 64 * 73    # 4672 — total actions in AlphaZero move encoding
TEMPERATURE  = 1.0        # softmax temperature for policy targets
                          #   1.0 = use raw score differences as-is
                          #   < 1.0 = sharper (best move dominates more)
                          #   > 1.0 = flatter (moves become more equal)

INPUT_CSV  = "dataset_with_puzzles.csv"    # output of generate_dataset_lichess.py
OUTPUT_NPZ = "dataset.npz"       # output consumed by train.py


# =========================
# build_soft_policy()
# =========================

def build_soft_policy(moves_uci, scores_cp, board):
    """
    Converts Stockfish's top-N moves and their centipawn scores into a sparse
    probability distribution over the 4672-action space.

    WHY SOFT TARGETS:
        With hard (one-hot) targets, the network learns only which move is best
        and gets zero gradient signal about 2nd or 3rd best moves. With soft
        targets, the network learns the full ranking — it knows that move B is
        almost as good as move A, and move C is far worse than both.

    HOW SOFTMAX WORKS HERE:
        1. Scores are from White's perspective (Stockfish convention).
           If it is Black's turn, negate all scores to get Black's perspective.
        2. Divide by (400 * temperature) to scale into a range where exp() is
           numerically stable and meaningful.
           400cp ≈ one pawn advantage → becomes exp(1) ≈ 2.7x more likely.
        3. Subtract the max score before exp() — this is a standard numerical
           stability trick that prevents overflow without changing the result.
        4. Divide by the sum so probabilities add to 1.

    SPARSE STORAGE:
        Instead of storing a 4672-float vector (mostly zeros), we store only
        the MULTIPV nonzero entries as (index, probability) pairs.
        Unused slots are padded with index=-1, prob=0.0.
        The training loop reconstructs the dense vector with scatter_().

    Args:
        moves_uci:  list of UCI move strings, length MULTIPV (may contain empty strings)
        scores_cp:  list of centipawn scores (from White's POV), length MULTIPV (may contain NaN)
        board:      chess.Board of the current position (used for legality check + turn)

    Returns:
        (indices, probs) tuple:
            indices: np.int16 array of shape (MULTIPV,), move indices (-1 for empty slots)
            probs:   np.float32 array of shape (MULTIPV,), probabilities (0.0 for empty slots)
        Returns (None, None) if no valid moves found.
    """

    pairs = []

    for uci, score in zip(moves_uci, scores_cp):

        # Skip empty or NaN entries (can occur if Stockfish found fewer than MULTIPV moves)
        if not uci or pd.isna(score):
            continue

        # Parse UCI string to move object
        try:
            move = chess.Move.from_uci(uci)
        except Exception:
            continue

        # Verify the move is legal in this position
        # (should always be true for valid Stockfish output, but defensive check)
        if move not in board.legal_moves:
            continue

        # Convert move to action space index
        idx = move_to_index(move, board)
        if idx is None:
            continue

        pairs.append((idx, float(score)))

    if not pairs:
        return None, None

    # ---- Orient scores to current player ----
    # Stockfish always reports from White's perspective.
    # If it is Black's turn, a score of +80 means Black is LOSING.
    # Negate to get "higher = better for current player" convention.
    if board.turn == chess.BLACK:
        pairs = [(idx, -score) for idx, score in pairs]

    # ---- Temperature-scaled softmax ----
    raw_scores = np.array([s for _, s in pairs], dtype=np.float64)
    raw_scores = raw_scores / (400.0 * TEMPERATURE)
    raw_scores -= raw_scores.max()      # numerical stability: subtract max before exp
    probs = np.exp(raw_scores)
    probs /= probs.sum()                # normalise to sum to 1

    # ---- Pack into fixed-length sparse arrays ----
    indices  = np.full(MULTIPV, -1,  dtype=np.int16)    # -1 = unused slot
    prob_arr = np.zeros(MULTIPV,     dtype=np.float32)

    for j, ((idx, _), prob) in enumerate(zip(pairs, probs)):
        indices[j]  = idx
        prob_arr[j] = float(prob)

    return indices, prob_arr


# =========================
# Load CSV
# =========================
#
# The CSV was produced by generate_dataset_lichess.py and has these columns:
#   FEN          — board position as a FEN string
#   Evaluation   — Stockfish centipawn score from White's perspective
#   BestMove     — best move in UCI notation (e.g. "e2e4")
#   Move1-Move5  — top-5 moves in UCI notation
#   Score1-Score5 — centipawn scores for those top-5 moves

print("Loading CSV ...")
data = pd.read_csv(INPUT_CSV)

fens        = data["FEN"].values
evaluations = data["Evaluation"].values

# Dynamically find MultiPV columns — handles datasets with fewer than MULTIPV columns
move_cols  = [f"Move{i+1}"  for i in range(MULTIPV)]
score_cols = [f"Score{i+1}" for i in range(MULTIPV)]
move_cols  = [c for c in move_cols  if c in data.columns]
score_cols = [c for c in score_cols if c in data.columns]

print(f"Loaded {len(data):,} rows. MultiPV columns found: {len(move_cols)}")


# =========================
# Convert Positions
# =========================
#
# For each row in the CSV:
#   1. Parse the FEN → board object
#   2. Build soft policy from top-5 moves + scores
#   3. Encode board state as 18-plane tensor
#   4. Map centipawn evaluation to (-1, 1) using tanh
#
# Skipped rows:
#   - Missing evaluation (NaN)
#   - All MultiPV moves invalid or illegal (and no valid BestMove fallback)
#   - move_to_index returns None for all moves (encoding failure)

states          = []
policy_indices  = []
policy_probs    = []
values          = []
skipped         = 0

for i, (fen, eval_score) in enumerate(zip(fens, evaluations)):

    if i % 10000 == 0:
        print(f"Processing {i:,} / {len(fens):,} ...")

    # ---- Value target ----
    if pd.isna(eval_score):
        skipped += 1
        continue

    board = chess.Board(fen)

    # ---- Soft policy target ----
    moves_uci  = [data[c].iloc[i] for c in move_cols]
    scores_raw = [data[c].iloc[i] for c in score_cols]

    # Safely convert score strings to float — replace unparseable with NaN
    scores_cp = []
    for s in scores_raw:
        try:
            scores_cp.append(float(s))
        except (ValueError, TypeError):
            scores_cp.append(float("nan"))

    indices, probs = build_soft_policy(moves_uci, scores_cp, board)

    # ---- Fallback to hard one-hot target ----
    # If MultiPV data is missing or all moves failed validation,
    # fall back to a one-hot target on the single best move.
    # This ensures we don't discard positions just because MultiPV failed.
    if indices is None:
        best_move_uci = data["BestMove"].iloc[i] if "BestMove" in data.columns else None
        if best_move_uci and not pd.isna(best_move_uci):
            try:
                move = chess.Move.from_uci(str(best_move_uci))
                if move in board.legal_moves:
                    idx = move_to_index(move, board)
                    if idx is not None:
                        indices    = np.full(MULTIPV, -1,  dtype=np.int16)
                        probs      = np.zeros(MULTIPV,     dtype=np.float32)
                        indices[0] = idx
                        probs[0]   = 1.0
            except Exception:
                pass

    if indices is None:
        skipped += 1
        continue

    # ---- Board tensor ----
    # fen_to_tensor returns shape (18, 8, 8) — 12 piece planes + 6 auxiliary planes
    state = fen_to_tensor(fen)

    # ---- Value target ----
    # Map centipawn score to (-1, 1) using tanh.
    #
    # WHY tanh instead of linear clip:
    #   Linear: +1000cp and +100cp both map to different but large values.
    #   tanh:   +1000cp → ~0.999, +100cp → ~0.245, 0cp → 0.0
    #   The tanh curve has diminishing returns for large advantages, which
    #   matches chess reality — being up 3 pawns vs 5 pawns is not very
    #   different strategically (both are winning). This makes the value
    #   head's learning task much easier.
    #
    # Dividing by 400 before tanh calibrates the curve so that:
    #   100cp (≈ 1 pawn)  → tanh(0.25) ≈ 0.24
    #   400cp (≈ 4 pawns) → tanh(1.0)  ≈ 0.76
    #   800cp (≈ 8 pawns) → tanh(2.0)  ≈ 0.96
    #
    # Value is stored from White's perspective (as Stockfish reports it).
    # During MCTS, it is negated when propagating to Black's nodes.

    value = float(np.tanh(eval_score / 400.0))

    states.append(state)
    policy_indices.append(indices)
    policy_probs.append(probs)
    values.append(value)


# =========================
# Save as Compressed NPZ
# =========================
#
# npz is numpy's multi-array archive format.
# savez_compressed applies zlib compression — typically 3-5x size reduction
# for int8 arrays (piece planes are mostly zeros).
#
# Array shapes and sizes for 500k positions:
#   states:          (500k, 18, 8, 8) int8    ≈  330 MB uncompressed, ~80 MB compressed
#   policy_indices:  (500k, 5)        int16   ≈    3 MB
#   policy_probs:    (500k, 5)        float32 ≈    6 MB
#   values:          (500k,)          float32 ≈    1 MB
#   Total compressed:                         ≈  ~90-100 MB

states_arr = np.stack(states).astype(np.int8)
p_indices  = np.stack(policy_indices).astype(np.int16)
p_probs    = np.stack(policy_probs).astype(np.float32)
values_arr = np.array(values, dtype=np.float32)

np.savez_compressed(
    OUTPUT_NPZ,
    states         = states_arr,
    policy_indices = p_indices,
    policy_probs   = p_probs,
    values         = values_arr
)

print("\nDataset conversion complete.")
print(f"Positions       : {len(states_arr):,}")
print(f"Skipped         : {skipped:,}  ({skipped / max(len(data), 1) * 100:.1f}%)")
print(f"States          : {states_arr.shape}  ({states_arr.nbytes / 1e6:.1f} MB uncompressed)")
print(f"Policy indices  : {p_indices.shape}   ({p_indices.nbytes / 1e6:.1f} MB)")
print(f"Policy probs    : {p_probs.shape}     ({p_probs.nbytes / 1e6:.1f} MB)")
print(f"Values          : {values_arr.shape}")
print(f"Saved to        : {OUTPUT_NPZ}")