"""
Lichess Dataset Generator
=========================
- Streams a partial download (~300MB) from a Lichess monthly PGN file
- Filters to games where both players are ≥1800 Elo
- Randomly samples positions from each game (natural distribution)
- Evaluates each position with Stockfish at depth=15, multipv=5
- Supports pause/resume: safe to Ctrl+C at any time and re-run
- Outputs a CSV in the same format as the previous generator

Requirements:
    pip install zstandard chess

Usage:
    python generate_dataset_lichess.py

To resume after interruption:
    python generate_dataset_lichess.py   (automatically detects checkpoint)
"""

import chess
import chess.engine
import chess.pgn
import csv
import io 
import json
import os
import random
import urllib.request
from pathlib import Path

import zstandard as zstd


# =========================
# CONFIGURATION
# =========================

STOCKFISH_PATH  = "/Users/frankzhang/Downloads/stockfish 2/stockfish-macos-x86-64-bmi2"

# Lichess monthly PGN file — we stream only the first DOWNLOAD_BYTES bytes
# January 2024 standard rated games
LICHESS_URL     = "https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst"
DOWNLOAD_BYTES  = 500 * 1024 * 1024   # 300 MB of compressed data

TARGET_POSITIONS  = 600000
POSITIONS_PER_GAME = 4               # random positions sampled per accepted game
MIN_ELO           = 2000             # both players must meet this threshold
MIN_TIME_CONTROL  = 180               # seconds — filters out bullet (< 3 min)
SKIP_OPENING_PLIES = 6                # skip the first N plies of each game (book moves)

DEPTH   = 12
MULTIPV = 5

OUTPUT_CSV        = "dataset_1M.csv"
CHECKPOINT_FILE   = "dataset_1M_checkpoint.json"
PARTIAL_PGN_CACHE = "lichess_partial.pgn.zst"  # cached compressed download

# Stockfish config — single engine, no parallelism needed here since
# bottleneck is network download + PGN parsing, not analysis
THREADS = 2
HASH_MB = 512


# =========================
# CHECKPOINT HELPERS
# =========================

def load_checkpoint():
    """Returns checkpoint dict or None if no checkpoint exists."""
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, "r") as f:
            cp = json.load(f)
        print(f"Resuming from checkpoint: {cp['positions_written']:,} positions already written.")
        return cp
    return None


def save_checkpoint(positions_written, games_processed, games_accepted):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "positions_written": positions_written,
            "games_processed":   games_processed,
            "games_accepted":    games_accepted,
        }, f)


def clear_checkpoint():
    if Path(CHECKPOINT_FILE).exists():
        os.remove(CHECKPOINT_FILE)


# =========================
# PARTIAL DOWNLOAD
# =========================

def download_partial(url, dest_path, max_bytes):
    """
    Downloads the first max_bytes of a URL to dest_path.
    Skips download if file already exists and is large enough.
    """
    dest = Path(dest_path)

    if dest.exists() and dest.stat().st_size >= max_bytes * 0.95:
        print(f"Using cached compressed file: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
        return

    print(f"Downloading first {max_bytes / 1e6:.0f} MB from Lichess...")
    print(f"URL: {url}")

    req = urllib.request.Request(
        url,
        headers={"Range": f"bytes=0-{max_bytes - 1}"}
    )

    downloaded = 0
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out:
        while True:
            chunk = response.read(1024 * 1024)   # 1MB chunks
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            mb = downloaded / 1e6
            if int(mb) % 10 == 0 and downloaded % (10 * 1024 * 1024) < 1024 * 1024:
                print(f"  Downloaded {mb:.0f} / {max_bytes / 1e6:.0f} MB ...")

    print(f"Download complete: {downloaded / 1e6:.1f} MB saved to {dest_path}")


# =========================
# GAME FILTERING
# =========================

def parse_time_control(tc_str):
    """
    Returns base time in seconds from a time control string like '600+5' or '300'.
    Returns 0 if unparseable.
    """
    if not tc_str or tc_str == "-":
        return 0
    try:
        base = tc_str.split("+")[0]
        return int(base)
    except (ValueError, IndexError):
        return 0


def game_passes_filter(game):
    """Returns True if game meets quality thresholds."""
    headers = game.headers

    # Must be a rated game
    if headers.get("Event", "").lower().find("rated") == -1:
        return False

    # Time control filter
    tc = parse_time_control(headers.get("TimeControl", ""))
    if tc < MIN_TIME_CONTROL:
        return False

    # Elo filter — both players
    try:
        white_elo = int(headers.get("WhiteElo", "0"))
        black_elo = int(headers.get("BlackElo", "0"))
    except ValueError:
        return False

    if white_elo < MIN_ELO or black_elo < MIN_ELO:
        return False

    # Game must have a result (not ongoing)
    result = headers.get("Result", "*")
    if result == "*":
        return False

    return True


# =========================
# POSITION SAMPLING
# =========================

def sample_positions_from_game(game, rng, n_samples):
    """
    Plays through a game and returns a list of randomly sampled board positions.
    Skips the first SKIP_OPENING_PLIES plies to avoid pure book moves.
    Returns list of chess.Board objects.
    """
    boards = []
    board  = game.board()

    for i, move in enumerate(game.mainline_moves()):
        board.push(move)
        # Skip opening plies and terminal positions
        if i < SKIP_OPENING_PLIES:
            continue
        if board.is_game_over():
            break
        boards.append(board.copy())

    if not boards:
        return []

    # Randomly sample without replacement (or all if fewer than n_samples)
    k = min(n_samples, len(boards))
    return rng.sample(boards, k)


# =========================
# STOCKFISH ANALYSIS
# =========================

def analyse_position(engine, board):
    """
    Returns (score_cp, multipv_results) or (None, None).
    score_cp is centipawns from White's perspective.
    """
    try:
        infos = engine.analyse(
            board,
            chess.engine.Limit(depth=DEPTH),
            multipv=MULTIPV
        )
    except Exception:
        return None, None

    if not infos:
        return None, None

    if not isinstance(infos, list):
        infos = [infos]

    top       = infos[0]
    score_obj = top["score"].white()
    score_cp  = score_obj.score(mate_score=10000)

    # Skip positions that are already won/lost or involve forced mate
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
            "move":  info["pv"][0].uci(),
            "score": s
        })

    if not results:
        return None, None

    return score_cp, results


def format_row(fen, score_cp, multipv_results):
    best_move = multipv_results[0]["move"]
    moves     = [r["move"]       for r in multipv_results]
    scores    = [str(r["score"]) for r in multipv_results]

    while len(moves)  < MULTIPV: moves.append("")
    while len(scores) < MULTIPV: scores.append("")

    return [fen, score_cp, best_move] + moves + scores


# =========================
# MAIN GENERATOR
# =========================

def main():
    rng = random.Random(42)

    # --- load checkpoint ---
    checkpoint        = load_checkpoint()
    positions_written = checkpoint["positions_written"] if checkpoint else 0
    games_processed   = checkpoint["games_processed"]   if checkpoint else 0
    games_accepted    = checkpoint["games_accepted"]     if checkpoint else 0

    if positions_written >= TARGET_POSITIONS:
        print(f"Already have {positions_written:,} positions. Nothing to do.")
        return

    # --- download partial PGN if needed ---
    download_partial(LICHESS_URL, PARTIAL_PGN_CACHE, DOWNLOAD_BYTES)

    # --- open Stockfish ---
    print("Starting Stockfish...")
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": THREADS, "Hash": HASH_MB})

    # --- build CSV header ---
    header = (
        ["FEN", "Evaluation", "BestMove"]
        + [f"Move{i+1}"  for i in range(MULTIPV)]
        + [f"Score{i+1}" for i in range(MULTIPV)]
    )

    # Open CSV in append mode if resuming, write mode if fresh
    csv_mode = "a" if checkpoint else "w"

    print(f"\nStarting generation. Target: {TARGET_POSITIONS:,} positions.")
    print(f"Already written: {positions_written:,}. Remaining: {TARGET_POSITIONS - positions_written:,}")
    print("Press Ctrl+C at any time to pause — progress is saved every 1000 positions.\n")

    try:
        with open(OUTPUT_CSV, csv_mode, newline="") as csv_file:
            writer = csv.writer(csv_file)

            # Write header only on fresh start
            if not checkpoint:
                writer.writerow(header)

            # --- stream and decompress PGN ---
            with open(PARTIAL_PGN_CACHE, "rb") as compressed_file:
                dctx        = zstd.ZstdDecompressor()
                stream      = dctx.stream_reader(compressed_file)
                text_stream = io.TextIOWrapper(stream, encoding="utf-8")

                # Skip games already processed in a previous run
                games_skipped = 0
                while games_skipped < games_processed:
                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        print("Warning: ran out of games while seeking to checkpoint position.")
                        break
                    games_skipped += 1

                if games_skipped > 0:
                    print(f"Skipped {games_skipped:,} already-processed games, resuming from game {games_processed + 1:,}.")

                # --- main loop ---
                while positions_written < TARGET_POSITIONS:

                    game = chess.pgn.read_game(text_stream)

                    if game is None:
                        print("\nRan out of games in the downloaded file.")
                        print("Consider increasing DOWNLOAD_BYTES or using a larger monthly file.")
                        break

                    games_processed += 1

                    # Apply quality filter
                    if not game_passes_filter(game):
                        continue

                    games_accepted += 1

                    # Sample positions from this game
                    sampled_boards = sample_positions_from_game(game, rng, POSITIONS_PER_GAME)

                    for board in sampled_boards:

                        if positions_written >= TARGET_POSITIONS:
                            break

                        score_cp, multipv_results = analyse_position(engine, board)

                        if score_cp is None:
                            continue

                        row = format_row(board.fen(), score_cp, multipv_results)
                        writer.writerow(row)
                        positions_written += 1

                        # Progress + checkpoint every 1000 positions
                        if positions_written % 1000 == 0:
                            pct = positions_written / TARGET_POSITIONS * 100
                            print(
                                f"Positions: {positions_written:,}/{TARGET_POSITIONS:,} ({pct:.1f}%) | "
                                f"Games processed: {games_processed:,} | "
                                f"Games accepted: {games_accepted:,} "
                                f"({games_accepted/max(games_processed,1)*100:.1f}% pass rate)"
                            )
                            csv_file.flush()
                            save_checkpoint(positions_written, games_processed, games_accepted)

    except KeyboardInterrupt:
        # Clean pause — checkpoint already saved at last 1000-position boundary
        print(f"\n\nPaused. Progress saved: {positions_written:,} positions written.")
        print(f"Re-run this script to continue from where you left off.")
        save_checkpoint(positions_written, games_processed, games_accepted)
        engine.quit()
        return

    engine.quit()

    if positions_written >= TARGET_POSITIONS:
        clear_checkpoint()
        print(f"\nDataset generation complete!")
        print(f"Total positions : {positions_written:,}")
        print(f"Games processed : {games_processed:,}")
        print(f"Games accepted  : {games_accepted:,}")
        print(f"Saved to        : {OUTPUT_CSV}")
    else:
        save_checkpoint(positions_written, games_processed, games_accepted)


if __name__ == "__main__":
    main()