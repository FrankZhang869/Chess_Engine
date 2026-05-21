"""import chess
import chess.engine
import csv

STOCKFISH_PATH = "/Users/frankzhang/Downloads/stockfish 2/stockfish-macos-x86-64-bmi2"

OPENING_TARGET = 100000
MIDGAME_TARGET = 100000
ENDGAME_TARGET = 100000

engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
engine.configure({"Threads": 8, "Hash": 2048})


def material_count(board):
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }

    total = 0
    for piece in board.piece_map().values():
        if piece.piece_type != chess.KING:
            total += values[piece.piece_type]

    return total


with open("dataset_300k.csv", "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["FEN", "Evaluation", "BestMove"])

    # ========================
    # OPENING POSITIONS
    # ========================

    opening_count = 0

    while opening_count < OPENING_TARGET:

        board = chess.Board()

        while not board.is_game_over() and board.fullmove_number <= 10:

            info = engine.analyse(board, chess.engine.Limit(time=0.02))
            score = info["score"].white().score(mate_score=10000)

            if score is not None and abs(score) < 1000:

                best_move = info["pv"][0].uci()
                writer.writerow([board.fen(), score, best_move])

                opening_count += 1

                if opening_count % 1000 == 0:
                    print(f"Opening positions: {opening_count}")
                    f.flush()

                if opening_count >= OPENING_TARGET:
                    break

            move = engine.play(board, chess.engine.Limit(time=0.01)).move
            board.push(move)


    # ========================
    # MIDGAME POSITIONS
    # ========================

    midgame_count = 0

    while midgame_count < MIDGAME_TARGET:

        board = chess.Board()

        while not board.is_game_over():

            if 10 < board.fullmove_number <= 30:

                info = engine.analyse(board, chess.engine.Limit(time=0.02))
                score = info["score"].white().score(mate_score=10000)

                if score is not None and abs(score) < 1000:

                    best_move = info["pv"][0].uci()
                    writer.writerow([board.fen(), score, best_move])

                    midgame_count += 1

                    if midgame_count % 1000 == 0:
                        print(f"Midgame positions: {midgame_count}")
                        f.flush()

                    if midgame_count >= MIDGAME_TARGET:
                        break

            if board.fullmove_number > 30:
                break

            move = engine.play(board, chess.engine.Limit(time=0.01)).move
            board.push(move)


    # ========================
    # ENDGAME POSITIONS
    # ========================

    endgame_count = 0

    while endgame_count < ENDGAME_TARGET:

        board = chess.Board()

        while not board.is_game_over():

            material = material_count(board)

            if material <= 14:

                info = engine.analyse(board, chess.engine.Limit(time=0.02))
                score = info["score"].white().score(mate_score=10000)

                if score is not None and abs(score) < 1000:

                    best_move = info["pv"][0].uci()
                    writer.writerow([board.fen(), score, best_move])

                    endgame_count += 1

                    if endgame_count % 1000 == 0:
                        print(f"Endgame positions: {endgame_count}")
                        f.flush()

                    if endgame_count >= ENDGAME_TARGET:
                        break

            if board.fullmove_number > 80:
                break

            move = engine.play(board, chess.engine.Limit(time=0.01)).move
            board.push(move)


engine.quit()

print("300k dataset generation complete!")"""

# new dataset generator (still 300k positions):
import chess
import chess.engine
import csv
import os
import random
import numpy as np
import multiprocessing as mp
from pathlib import Path

# =========================
# CONFIGURATION
# =========================

STOCKFISH_PATH = "/Users/frankzhang/Downloads/stockfish 2/stockfish-macos-x86-64-bmi2"

TOTAL_POSITIONS   = 300_000   # total across all workers
NUM_WORKERS       = 4         # set to number of logical cores / 2
THREADS_PER_ENGINE = 1        # 4 workers x 1 thread = 4 total, fits 4-core Mac exactly

DEPTH             = 12        # analysis depth per position
MULTIPV           = 5         # top N moves for soft policy targets

# Phase targets (fractions of TOTAL_POSITIONS)
OPENING_FRAC  = 0.33
MIDGAME_FRAC  = 0.34
ENDGAME_FRAC  = 0.33

# How many random moves to play at the start of each game to diversify openings
# Each worker also uses its own random seed so games diverge further
MIN_RANDOM_PLIES = 0
MAX_RANDOM_PLIES = 4

OUTPUT_DIR = Path("dataset_parts")
FINAL_OUTPUT = "dataset_300k.csv"


# =========================
# OPENING BOOK (optional extra diversity)
# A small set of common opening moves to seed variety before random play
# =========================

SEED_OPENINGS = [
    ["e2e4"],
    ["d2d4"],
    ["c2c4"],
    ["g1f3"],
    ["e2e4", "e7e5"],
    ["e2e4", "c7c5"],
    ["e2e4", "e7e6"],
    ["d2d4", "d7d5"],
    ["d2d4", "g8f6"],
    ["c2c4", "e7e5"],
    ["e2e4", "e7e5", "g1f3"],
    ["d2d4", "d7d5", "c2c4"],
]


def material_count(board):
    values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9
    }
    return sum(
        values[p.piece_type]
        for p in board.piece_map().values()
        if p.piece_type != chess.KING
    )


def random_opening(board, engine, rng):
    """
    Play a random seed opening + random legal moves to diversify starting position.
    Returns the board after the randomized opening.
    """
    # Pick a random seed opening line
    seed = rng.choice(SEED_OPENINGS)
    for uci in seed:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            board.push(move)
        else:
            break

    # Then play N additional random legal moves
    n_random = rng.randint(MIN_RANDOM_PLIES, MAX_RANDOM_PLIES)
    for _ in range(n_random):
        if board.is_game_over():
            break
        legal = list(board.legal_moves)
        board.push(rng.choice(legal))

    return board


def analyse_position(engine, board, depth, multipv):
    """
    Returns (score_cp, multipv_results) or (None, None) on failure.
    score_cp is from White's perspective.
    multipv_results is a list of dicts: [{"move": uci, "score": cp}, ...]
    """
    try:
        infos = engine.analyse(
            board,
            chess.engine.Limit(depth=depth),
            multipv=multipv
        )
    except Exception:
        return None, None

    if not infos:
        return None, None

    # infos is a list when multipv > 1
    if not isinstance(infos, list):
        infos = [infos]

    top = infos[0]
    score_obj = top["score"].white()
    score_cp = score_obj.score(mate_score=10000)

    if score_cp is None or abs(score_cp) >= 1000:
        return None, None

    results = []
    for info in infos:
        if "pv" not in info or not info["pv"]:
            continue
        s = info["score"].white().score(mate_score=10000)
        if s is None:
            continue
        results.append({
            "move": info["pv"][0].uci(),
            "score": s
        })

    return score_cp, results


def format_row(fen, score_cp, multipv_results):
    """
    Build a CSV row.
    Columns: FEN, Evaluation, BestMove, Move1..MoveN, Score1..ScoreN
    where Move/Score pairs are the top-N soft policy moves.
    """
    best_move = multipv_results[0]["move"] if multipv_results else ""

    moves  = [r["move"]  for r in multipv_results]
    scores = [str(r["score"]) for r in multipv_results]

    # Pad to MULTIPV length so CSV columns are consistent
    while len(moves)  < MULTIPV: moves.append("")
    while len(scores) < MULTIPV: scores.append("")

    return [fen, score_cp, best_move] + moves + scores


# =========================
# WORKER FUNCTION
# =========================

def worker(worker_id, positions_per_worker, output_path, seed):

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": THREADS_PER_ENGINE, "Hash": 128})

    opening_target  = int(positions_per_worker * OPENING_FRAC)
    midgame_target  = int(positions_per_worker * MIDGAME_FRAC)
    endgame_target  = positions_per_worker - opening_target - midgame_target

    header = (
        ["FEN", "Evaluation", "BestMove"]
        + [f"Move{i+1}"  for i in range(MULTIPV)]
        + [f"Score{i+1}" for i in range(MULTIPV)]
    )

    collected = {"opening": 0, "midgame": 0, "endgame": 0}
    targets   = {"opening": opening_target, "midgame": midgame_target, "endgame": endgame_target}

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        while any(collected[k] < targets[k] for k in collected):

            # --- start a new game with a randomized opening ---
            board = chess.Board()
            board = random_opening(board, engine, rng)

            if board.is_game_over():
                continue

            # --- play the game, collecting positions ---
            while not board.is_game_over() and board.fullmove_number <= 120:

                move_num = board.fullmove_number
                material = material_count(board)

                # Determine phase
                # Widened opening window to move 15 to compensate for
                # seed opening moves consuming early fullmove numbers
                if move_num <= 15:
                    phase = "opening"
                elif move_num <= 35:
                    phase = "midgame"
                elif material <= 14:
                    phase = "endgame"
                else:
                    phase = None

                if phase and collected[phase] < targets[phase]:

                    score_cp, multipv_results = analyse_position(
                        engine, board, DEPTH, MULTIPV
                    )

                    if score_cp is not None and multipv_results:
                        row = format_row(board.fen(), score_cp, multipv_results)
                        writer.writerow(row)
                        collected[phase] += 1

                        total = sum(collected.values())
                        if total % 1000 == 0:
                            print(
                                f"[Worker {worker_id}] "
                                f"opening={collected['opening']}/{targets['opening']} "
                                f"midgame={collected['midgame']}/{targets['midgame']} "
                                f"endgame={collected['endgame']}/{targets['endgame']}"
                            )
                            f.flush()

                    if all(collected[k] >= targets[k] for k in collected):
                        break

                # Advance the game: use best move from analysis if we just analysed,
                # otherwise ask Stockfish quickly to keep the game moving
                if phase and multipv_results:
                    next_move = chess.Move.from_uci(multipv_results[0]["move"])
                else:
                    try:
                        result = engine.play(board, chess.engine.Limit(depth=8))
                        next_move = result.move
                    except Exception:
                        break

                if next_move in board.legal_moves:
                    board.push(next_move)
                else:
                    break

    engine.quit()
    print(f"[Worker {worker_id}] Done. Wrote to {output_path}")


# =========================
# MERGE PARTIAL CSVs
# =========================

def merge_parts(output_dir, final_output, num_workers):
    print(f"\nMerging {num_workers} partial files into {final_output} ...")
    header_written = False

    with open(final_output, "w", newline="") as fout:
        writer = csv.writer(fout)

        for i in range(num_workers):
            part_path = output_dir / f"part_{i}.csv"
            with open(part_path, "r", newline="") as fin:
                reader = csv.reader(fin)
                for j, row in enumerate(reader):
                    if j == 0:  # header
                        if not header_written:
                            writer.writerow(row)
                            header_written = True
                        continue
                    writer.writerow(row)

    print(f"Merged dataset saved to {final_output}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    OUTPUT_DIR.mkdir(exist_ok=True)

    positions_per_worker = TOTAL_POSITIONS // NUM_WORKERS
    # Give remainder to last worker
    positions_list = [positions_per_worker] * NUM_WORKERS
    positions_list[-1] += TOTAL_POSITIONS - sum(positions_list)

    # Each worker gets a unique seed so their random openings diverge
    seeds = [42 + i * 1000 for i in range(NUM_WORKERS)]

    processes = []
    for i in range(NUM_WORKERS):
        output_path = OUTPUT_DIR / f"part_{i}.csv"
        p = mp.Process(
            target=worker,
            args=(i, positions_list[i], output_path, seeds[i])
        )
        processes.append(p)
        p.start()
        print(f"Started worker {i} (seed={seeds[i]}, positions={positions_list[i]})")

    for p in processes:
        p.join()

    merge_parts(OUTPUT_DIR, FINAL_OUTPUT, NUM_WORKERS)
    print("\nDataset generation complete.")
    print(f"Output: {FINAL_OUTPUT}")