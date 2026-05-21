import pandas as pd
import chess
import numpy as np

df = pd.read_csv("/Users/frankzhang/Downloads/lichess_db_puzzle.csv",
                 nrows=50000,   # read first 50k, filter down to 10k
                 names=["PuzzleId","FEN","Moves","Rating",
                        "RatingDeviation","Popularity","NbPlays",
                        "Themes","GameUrl","OpeningTags"])

df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
# Filter for mate puzzles at appropriate difficulty
mate_puzzles = df[
    df["Themes"].str.contains("mateIn1|mateIn2|mateIn3", na=False) &
    (df["Rating"] >= 800) &    # not too easy
    (df["Rating"] <= 1800)     # not too hard for current engine level
].head(10000)

print(f"Filtered to {len(mate_puzzles)} puzzles")

rows = []

for _, row in mate_puzzles.iterrows():
    try:
        board = chess.Board(row["FEN"])
        moves = row["Moves"].split()

        # Push opponent's move to reach puzzle position
        board.push_uci(moves[0])

        # The solution move
        solution_uci = moves[1]
        solution_move = chess.Move.from_uci(solution_uci)

        if solution_move not in board.legal_moves:
            continue

        # Use high centipawn value for mate positions
        # mateIn1 = 900cp, mateIn2 = 800cp, mateIn3 = 700cp
        themes = row["Themes"]
        if "mateIn1" in themes:
            eval_cp = 900
        elif "mateIn2" in themes:
            eval_cp = 800
        else:
            eval_cp = 700

        # Orient eval to current player
        if board.turn == chess.BLACK:
            eval_cp = -eval_cp

        rows.append({
            "FEN":       board.fen(),
            "Evaluation": eval_cp,
            "BestMove":  solution_uci,
            "Move1":     solution_uci,
            "Score1":    eval_cp,
            "Move2": "", "Score2": "",
            "Move3": "", "Score3": "",
            "Move4": "", "Score4": "",
            "Move5": "", "Score5": "",
        })

    except Exception:
        continue

puzzle_df = pd.DataFrame(rows)
puzzle_df.to_csv("puzzle_positions.csv", index=False)
print(f"Saved {len(puzzle_df)} puzzle positions")

main_df   = pd.read_csv("dataset_1M.csv")
puzzle_df = pd.read_csv("puzzle_positions.csv")

# Concatenate and shuffle
combined = pd.concat([main_df, puzzle_df], ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
combined.to_csv("dataset_with_puzzles.csv", index=False)

print(f"Combined dataset: {len(combined):,} positions")
print(f"  Lichess games: {len(main_df):,}")
print(f"  Puzzles:       {len(puzzle_df):,}")