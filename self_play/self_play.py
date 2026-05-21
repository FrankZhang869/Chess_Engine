"""
self_play.py
============
Generates self-play games using batched MCTS.

Batched MCTS runs N games simultaneously and groups their network
evaluation calls into a single batched forward pass. This is ~8x faster
than sequential MCTS on GPU because the GPU processes a batch of 8
positions in roughly the same time as a batch of 1.

How batched MCTS works:
    Each game maintains its own search tree and tracks which simulation
    it is currently running. At each step, all games simultaneously:
        1. Select a leaf node (pure CPU — no network needed)
        2. Check if the leaf needs network evaluation
        3. If yes → add to pending batch
        4. Once all games have a pending position → one batched network call
        5. Distribute results back to each game → backpropagate

Self-play targets:
    policy target: MCTS visit counts / total visits (not Stockfish MultiPV)
    value target:  actual game outcome +1/0/-1 (not Stockfish centipawn eval)

Temperature schedule:
    Moves 1-30: sample proportionally to visit counts (temperature=1.0)
                → adds diversity, prevents all games following same path
    Move 31+:   greedy selection (temperature=0)
                → best move always chosen in endgame
"""

import torch
import chess
import numpy as np
from collections import defaultdict

from src.encoder import fen_to_tensor, move_to_index
from self_play.replay_buffer import ReplayBuffer


# =========================
# Configuration
# =========================

SIMULATIONS_PER_MOVE  = 200     # MCTS simulations per move
PARALLEL_GAMES        = 8       # games running simultaneously
TEMPERATURE_THRESHOLD = 30      # moves before switching to greedy selection
C_PUCT                = 1.5     # exploration constant
DIRICHLET_ALPHA       = 0.3     # noise concentration (AlphaZero chess value)
DIRICHLET_WEIGHT      = 0.25    # noise mix at root (AlphaZero value)
MAX_GAME_LENGTH       = 200     # moves before declaring draw (prevents infinite games)
MULTIPV               = 5       # top moves stored in sparse policy


# =========================
# Single Game State
# =========================

class GameState:
    """
    Tracks all state for one self-play game.

    Maintains the board, the MCTS tree rooted at the current position,
    and the history of (state, policy, value_placeholder) tuples that
    will be assigned outcomes once the game ends.
    """

    def __init__(self, game_id):
        self.game_id   = game_id
        self.board     = chess.Board()
        self.root      = None          # current MCTS root node
        self.move_count = 0

        # Collected training data — value filled in after game ends
        self.states         = []       # (18,8,8) int8 arrays
        self.policy_indices = []       # (MULTIPV,) int16 arrays
        self.policy_probs   = []       # (MULTIPV,) float32 arrays
        self.current_players = []      # whose turn at each position (for value assignment)

        self.done   = False
        self.result = None             # "1-0", "0-1", "1/2-1/2"

    def is_done(self):
        return self.done


# =========================
# MCTS Node
# =========================

class Node:
    """Lightweight MCTS node for self-play."""

    def __init__(self, prior=0.0):
        self.prior     = prior
        self.visits    = 0
        self.value_sum = 0.0
        self.children  = {}        # move → Node
        self.expanded  = False

    def value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def select_child(self, c_puct):
        sqrt_visits = np.sqrt(self.visits + 1)
        best_score  = -float("inf")
        best_move   = None
        best_child  = None

        for move, child in self.children.items():
            q     = child.value()
            u     = c_puct * child.prior * sqrt_visits / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_move  = move
                best_child = child

        return best_move, best_child


# =========================
# Batched Self-Play Generator
# =========================

class BatchedSelfPlay:
    """
    Runs PARALLEL_GAMES games simultaneously, batching network calls.

    For each simulation step:
        1. All active games select a leaf node (CPU only)
        2. Terminal leaves get their value directly
        3. Non-terminal leaves are batched for network evaluation
        4. Network evaluates all pending positions in one forward pass
        5. Results distributed back, backpropagation runs for all games
    """

    def __init__(self, model, device):
        self.model  = model
        self.device = device

    @torch.no_grad()
    def evaluate_batch(self, boards):
        """
        Evaluate a batch of board positions with the network.

        Args:
            boards: list of chess.Board objects

        Returns:
            policies: list of numpy arrays (4672,) — softmaxed policy
            values:   list of floats — value in current-player perspective
        """
        if not boards:
            return [], []

        # Encode all boards into a batch tensor
        tensors = np.stack([fen_to_tensor(b.fen()) for b in boards]).astype(np.float32)
        batch   = torch.from_numpy(tensors).to(self.device)

        policy_logits, values = self.model(batch)

        # Softmax policies
        policies_np = torch.softmax(policy_logits, dim=1).cpu().numpy()

        # Values in current-player perspective
        values_np = values.cpu().numpy().flatten()
        result_values   = []
        result_policies = []

        for i, board in enumerate(boards):
            v = float(values_np[i])
            if board.turn == chess.BLACK:
                v = -v
            result_values.append(v)
            result_policies.append(policies_np[i])

        return result_policies, result_values

    def expand_node(self, node, board, policy):
        """
        Expand a node using the network's policy output.
        Creates child nodes for all legal moves with their prior probabilities.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            node.expanded = True
            return

        priors = []
        valid_moves = []

        for move in legal_moves:
            idx = move_to_index(move, board)
            if idx is None:
                continue
            valid_moves.append(move)
            priors.append(policy[idx])

        if not valid_moves:
            node.expanded = True
            return

        # Renormalise over legal moves
        priors = np.array(priors, dtype=np.float32)
        priors = priors / (priors.sum() + 1e-8)

        for move, prior in zip(valid_moves, priors):
            node.children[move] = Node(prior=float(prior))

        node.expanded = True

    def add_dirichlet_noise(self, node):
        """Add Dirichlet noise to root node priors for exploration."""
        if not node.children:
            return
        moves  = list(node.children.keys())
        noise  = np.random.dirichlet([DIRICHLET_ALPHA] * len(moves))
        for move, n in zip(moves, noise):
            node.children[move].prior = (
                (1 - DIRICHLET_WEIGHT) * node.children[move].prior
                + DIRICHLET_WEIGHT * n
            )

    def select_leaf(self, node, board):
        """
        Traverse tree from node following PUCT until reaching an unexpanded leaf.
        Returns (leaf_node, leaf_board, path) where path is list of (node, move) pairs.
        """
        path        = []
        current     = node
        current_board = board.copy()

        while current.expanded and current.children and not current_board.is_game_over():
            move, child = current.select_child(C_PUCT)
            path.append((current, move))
            current_board.push(move)
            current = child

        return current, current_board, path

    def backpropagate(self, path, leaf_node, value):
        """
        Update visit counts and value sums from leaf back to root.
        Value is negated at each level (alternating player perspective).
        """
        leaf_node.visits    += 1
        leaf_node.value_sum += value
        value = -value

        for node, move in reversed(path):
            child = node.children[move]
            child.visits    += 1
            child.value_sum += value
            value = -value

    def get_policy_target(self, root):
        """
        Convert MCTS visit counts to sparse policy target.

        Policy target = visit_count / total_visits for each child.
        This is richer than Stockfish's softmax because it reflects
        the engine's actual search experience at its current strength.

        Returns (indices, probs) sparse arrays of length MULTIPV.
        """
        if not root.children:
            indices  = np.full(MULTIPV, -1,  dtype=np.int16)
            probs    = np.zeros(MULTIPV,     dtype=np.float32)
            return indices, probs

        moves  = list(root.children.keys())
        visits = np.array([root.children[m].visits for m in moves], dtype=np.float32)
        total  = visits.sum()

        if total == 0:
            indices  = np.full(MULTIPV, -1,  dtype=np.int16)
            probs    = np.zeros(MULTIPV,     dtype=np.float32)
            return indices, probs

        probs_full = visits / total

        # Take top MULTIPV moves by visit count
        top_k   = min(MULTIPV, len(moves))
        top_idx = np.argsort(probs_full)[::-1][:top_k]

        indices  = np.full(MULTIPV, -1,  dtype=np.int16)
        probs    = np.zeros(MULTIPV,     dtype=np.float32)

        for j, i in enumerate(top_idx):
            move = moves[i]
            idx  = move_to_index(move, chess.Board(None))
            # Use a dummy board just for encoding — root board not accessible here
            # We'll re-encode properly in run_games using the actual board
            indices[j] = i   # store move list index temporarily
            probs[j]   = probs_full[i]

        return indices, probs

    def select_move(self, root, move_number):
        """
        Select move based on visit counts with temperature schedule.

        Moves 1-30:  sample proportionally (temperature=1.0) for diversity
        Move 31+:    greedy selection for best endgame play
        """
        moves  = list(root.children.keys())
        visits = np.array([root.children[m].visits for m in moves], dtype=np.float32)

        if move_number > TEMPERATURE_THRESHOLD or visits.sum() == 0:
            return moves[np.argmax(visits)]

        # Temperature=1.0: sample proportional to visit counts
        probs = visits / visits.sum()
        return moves[np.random.choice(len(moves), p=probs)]

    def run_games(self, num_games, replay_buffer):
        """
        Generate num_games self-play games using batched MCTS.

        For each move in each game:
            1. Run SIMULATIONS_PER_MOVE simulations using batched network calls
            2. Store (state, policy_target) for this position
            3. Select and play the move
            4. Repeat until game over

        Once game ends, assign game outcome as value target to all positions.
        Add all positions to replay buffer.

        Args:
            num_games:      number of games to generate
            replay_buffer:  ReplayBuffer to add positions to

        Returns:
            results: dict with win/draw/loss counts and avg game length
        """
        results       = {"white_wins": 0, "black_wins": 0, "draws": 0, "total_moves": 0}
        games_done    = 0
        games_started = 0

        # Process games in batches of PARALLEL_GAMES
        while games_done < num_games:

            # Start a new batch of games
            batch_size  = min(PARALLEL_GAMES, num_games - games_done)
            game_states = [GameState(games_started + i) for i in range(batch_size)]
            games_started += batch_size

            print(f"  Starting games {games_done+1}-{games_done+batch_size} of {num_games}")

            # Initialise root nodes for all games
            # First batch evaluation to set up roots
            boards  = [gs.board for gs in game_states]
            policies, values = self.evaluate_batch(boards)

            for gs, policy, value in zip(game_states, policies, values):
                gs.root = Node()
                self.expand_node(gs.root, gs.board, policy)
                self.add_dirichlet_noise(gs.root)
                self.backpropagate([], gs.root, value)

            # Play until all games in this batch are done
            while any(not gs.is_done() for gs in game_states):

                active = [gs for gs in game_states if not gs.is_done()]

                # Run SIMULATIONS_PER_MOVE simulations for each active game
                for sim in range(SIMULATIONS_PER_MOVE):

                    # Phase 1: Selection — find leaf for each game (CPU only)
                    pending_boards  = []   # boards needing network evaluation
                    pending_games   = []   # corresponding game states
                    pending_leaves  = []   # corresponding leaf nodes
                    pending_paths   = []   # paths for backprop

                    terminal_games  = []   # games with terminal leaves
                    terminal_values = []   # their values
                    terminal_paths  = []   # their paths

                    for gs in active:
                        leaf, leaf_board, path = self.select_leaf(gs.root, gs.board)

                        if leaf_board.is_game_over():
                            # Terminal node — get value directly without network
                            if leaf_board.is_checkmate():
                                value = -1.0    # current player is in checkmate
                            else:
                                value = 0.0     # draw
                            terminal_games.append(gs)
                            terminal_values.append(value)
                            terminal_paths.append((leaf, path))
                        elif not leaf.expanded:
                            # Needs network evaluation
                            pending_boards.append(leaf_board)
                            pending_games.append(gs)
                            pending_leaves.append(leaf)
                            pending_paths.append(path)
                        else:
                            # Already expanded but no children (shouldn't happen often)
                            terminal_games.append(gs)
                            terminal_values.append(0.0)
                            terminal_paths.append((leaf, path))

                    # Phase 2: Batched network evaluation for all pending leaves
                    if pending_boards:
                        policies, values = self.evaluate_batch(pending_boards)

                        for gs, policy, value, leaf, path in zip(
                            pending_games, policies, values, pending_leaves, pending_paths
                        ):
                            self.expand_node(leaf, pending_boards[pending_games.index(gs)], policy)
                            self.backpropagate(path, leaf, value)

                    # Phase 3: Backpropagate terminal nodes
                    for gs, value, (leaf, path) in zip(
                        terminal_games, terminal_values, terminal_paths
                    ):
                        self.backpropagate(path, leaf, value)

                # After all simulations: record position and select move for each active game
                for gs in active:

                    if gs.board.is_game_over():
                        gs.done   = True
                        gs.result = gs.board.result()
                        continue

                    # Record training data for this position
                    state = fen_to_tensor(gs.board.fen())

                    # Build sparse policy from visit counts
                    moves  = list(gs.root.children.keys())
                    visits = np.array([gs.root.children[m].visits for m in moves], dtype=np.float32)
                    total  = visits.sum()
                    probs_full = visits / (total + 1e-8)
                    top_k  = min(MULTIPV, len(moves))
                    top_idx = np.argsort(probs_full)[::-1][:top_k]

                    indices  = np.full(MULTIPV, -1,  dtype=np.int16)
                    probs    = np.zeros(MULTIPV,     dtype=np.float32)

                    for j, i in enumerate(top_idx):
                        move = moves[i]
                        idx  = move_to_index(move, gs.board)
                        if idx is not None:
                            indices[j] = idx
                            probs[j]   = probs_full[i]

                    gs.states.append(state)
                    gs.policy_indices.append(indices)
                    gs.policy_probs.append(probs)
                    gs.current_players.append(gs.board.turn)

                    # Select and play move
                    move = self.select_move(gs.root, gs.move_count)
                    gs.board.push(move)
                    gs.move_count += 1

                    # Check game over after move
                    if gs.board.is_game_over() or gs.move_count >= MAX_GAME_LENGTH:
                        gs.done   = True
                        gs.result = gs.board.result() if gs.board.is_game_over() else "1/2-1/2"
                        continue

                    # Re-root tree at played move for next iteration
                    # Reuse existing subtree if available (tree reuse)
                    if move in gs.root.children:
                        gs.root = gs.root.children[move]
                    else:
                        # Shouldn't happen but handle gracefully
                        gs.root = Node()
                        policy, value = self.evaluate_batch([gs.board])
                        self.expand_node(gs.root, gs.board, policy[0])
                        self.backpropagate([], gs.root, value[0])

                    self.add_dirichlet_noise(gs.root)

            # Games in this batch are done — assign outcomes and add to buffer
            for gs in game_states:
                result = gs.result

                if result == "1-0":
                    outcome = 1.0
                    results["white_wins"] += 1
                elif result == "0-1":
                    outcome = -1.0
                    results["black_wins"] += 1
                else:
                    outcome = 0.0
                    results["draws"] += 1

                results["total_moves"] += gs.move_count

                # Assign value from each position's current player's perspective
                values = []
                for player in gs.current_players:
                    if player == chess.WHITE:
                        values.append(outcome)
                    else:
                        values.append(-outcome)

                replay_buffer.add_game(
                    gs.states,
                    gs.policy_indices,
                    gs.policy_probs,
                    values
                )

                games_done += 1
                print(
                    f"  Game {games_done}/{num_games} done | "
                    f"Result: {result} | Moves: {gs.move_count}"
                )

        avg_length = results["total_moves"] / num_games
        print(f"\nSelf-play complete:")
        print(f"  White wins: {results['white_wins']}  "
              f"Black wins: {results['black_wins']}  "
              f"Draws: {results['draws']}")
        print(f"  Average game length: {avg_length:.1f} moves")

        return results