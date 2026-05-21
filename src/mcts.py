import torch
import chess
import numpy as np

from encoder import move_to_index
from encoder import fen_to_tensor


# =========================
# Node
# =========================

class Node:
    """
    Represents a single position in the MCTS search tree.

    Each node stores:
        board      — the chess position at this node
        parent     — the node we came from (None for root)
        prior      — the network's initial probability for the move that led here
        children   — dict mapping chess.Move → child Node
        visits     — how many simulations have passed through this node
        value_sum  — sum of all backed-up values through this node

    The average value (value_sum / visits) estimates how good this position
    is for the player who just moved. Higher = better for that player.
    """

    def __init__(self, board, parent=None, prior=0):
        self.board     = board
        self.parent    = parent
        self.prior     = prior
        self.children  = {}
        self.visits    = 0
        self.value_sum = 0

    def value(self):
        """
        Average value across all simulations through this node.
        Returns 0 if unvisited — treated as neutral during PUCT selection.
        """
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits


# =========================
# PUCT Selection
# =========================

def select_child(node, c_puct=1.5):
    """
    Selects the child with the highest PUCT score.

    PUCT = Q + U
        Q = child.value()
            Exploitation term — how good has this move been on average?
            Higher Q = this move has led to good positions in past simulations.

        U = c_puct * prior * sqrt(parent_visits) / (1 + child_visits)
            Exploration term — how much should we try this under-explored move?
            - prior:         network's initial confidence in this move
            - sqrt(parent):  grows as parent gets more visits, maintaining pressure to explore
            - 1+child_visits: decays as child gets visited, reducing bonus for well-explored moves

    c_puct controls the exploration/exploitation tradeoff:
        Higher c_puct → explore more broadly (visit low-confidence moves more)
        Lower c_puct  → exploit more aggressively (trust Q values more)
        1.5 is a standard starting value.

    Args:
        node:   the node whose children we are selecting among
        c_puct: exploration constant

    Returns:
        (best_move, best_child) tuple
    """
    best_score = -float("inf")
    best_move  = None
    best_child = None

    sqrt_parent_visits = np.sqrt(node.visits + 1)
    # +1 prevents sqrt(0) during first visit and keeps exploration bonus nonzero

    for move, child in node.children.items():

        Q = child.value()

        U = c_puct * child.prior * (
            sqrt_parent_visits / (1 + child.visits)
        )

        score = Q + U

        if score > best_score:
            best_score = score
            best_move  = move
            best_child = child

    return best_move, best_child


# =========================
# Expand Node
# =========================

def expand(node, model, device, add_noise=False):
    """
    Evaluates a leaf node using the neural network and creates its children.

    Steps:
        1. Check for terminal position (game over) — return outcome directly
        2. Encode board → 18-plane tensor
        3. Run network → policy logits + value
        4. Extract prior probabilities for all legal moves
        5. Guard against None indices from move_to_index
        6. Renormalise priors over legal moves only
        7. Optionally add Dirichlet noise at root
        8. Create child nodes for each legal move
        9. Return value in current-player perspective

    Value perspective:
        The network always sees the board mirrored to White-to-move (see fen_to_tensor).
        So the network's value output is always from White's perspective.
        We convert to current-player perspective here by negating for Black.
        backpropagate() then flips the value at each level going up the tree,
        converting between child perspective and parent perspective correctly.

    Args:
        node:      the leaf node to expand
        model:     the neural network
        device:    torch device (cpu or cuda)
        add_noise: if True, add Dirichlet noise to priors (root node only)

    Returns:
        value in current-player perspective, or -1/0 for terminal positions
    """

    board = node.board

    # ---- Terminal state handling ----
    # Return the game result directly without calling the network.
    # Checkmate: the player to move is in checkmate → they lose → return -1
    # Everything else (stalemate, draw by repetition, etc.) → return 0
    if board.is_game_over():
        if board.is_checkmate():
            return -1
        return 0

    # ---- Encode board ----
    state = fen_to_tensor(board.fen())
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    # unsqueeze(0) adds batch dimension: (18, 8, 8) → (1, 18, 8, 8)

    # ---- Network forward pass ----
    with torch.no_grad():
        # torch.no_grad() disables gradient tracking — not needed for inference
        # and saves significant memory + compute
        policy_logits, value = model(state)

    # Convert logits to probabilities over all 4672 actions
    policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]

    # ---- Extract legal move priors ----
    # The policy covers all 4672 possible actions including illegal ones.
    # We extract only the probabilities for legal moves in this position,
    # then renormalise so they sum to 1.
    legal_moves  = list(board.legal_moves)
    valid_moves  = []
    move_priors  = []

    for move in legal_moves:
        idx = move_to_index(move, board)

        # Guard: move_to_index returns None for moves it cannot encode.
        # This should be rare for legal moves but must be handled to prevent
        # policy[None] from crashing the search.
        if idx is None:
            continue

        valid_moves.append(move)
        move_priors.append(policy[idx])

    # If somehow all moves failed encoding, treat as draw
    if not valid_moves:
        return 0

    move_priors = np.array(move_priors, dtype=np.float32)

    # Renormalise over legal moves — raw policy mass was spread over 4672 actions
    total = np.sum(move_priors) + 1e-8
    move_priors /= total

    # ---- Dirichlet noise (root node only) ----
    # Adds randomness to the root's priors to ensure the search explores
    # moves the network rates as unlikely. Without this, MCTS can get stuck
    # always following the same deterministic path.
    # alpha=0.3 is the AlphaZero value for chess.
    # 75/25 split preserves most of the network's knowledge while adding noise.
    if add_noise and len(valid_moves) > 0:
        noise      = np.random.dirichlet([0.3] * len(move_priors))
        move_priors = 0.75 * move_priors + 0.25 * noise

    # ---- Create child nodes ----
    for move, prior in zip(valid_moves, move_priors):
        child_board = board.copy()
        child_board.push(move)

        node.children[move] = Node(
            child_board,
            parent = node,
            prior  = float(prior)
        )

    # ---- Convert value to current-player perspective ----
    # Network outputs value from White's perspective (board always mirrored to White).
    # Negate for Black so that higher value always means better for current player.
    # backpropagate() then handles the alternating perspective as it walks up the tree.
    value = value.item()
    if board.turn == chess.BLACK:
        value = -value

    return value


# =========================
# Backpropagation
# =========================

def backpropagate(node, value):
    """
    Walks from the expanded leaf back to the root, updating visit counts
    and value sums at every node along the path.

    Value is negated at each level because parent and child are opponents —
    a good position for the child is bad for the parent.

    Example for a 3-level path (root → A → B, where B was expanded):
        B gets value  v   (current player at B)
        A gets value -v   (A's player is B's opponent)
        root gets   +v   (root's player is A's opponent = same as B's player)

    Args:
        node:  the node that was just expanded (leaf of the simulation path)
        value: evaluation from that node's current-player perspective
    """
    while node is not None:
        node.visits    += 1
        node.value_sum += value
        value           = -value    # flip perspective for parent
        node            = node.parent


# =========================
# Run One Simulation
# =========================

def run_simulation(root, model, device):
    """
    Runs one full MCTS simulation: selection → expansion → backpropagation.

    Selection:
        Starting at root, repeatedly pick the child with the highest PUCT score
        until we reach a node with no children (unexpanded leaf).

    Expansion:
        Call the network on the leaf to get its value and create its children.

    Backpropagation:
        Walk back up the tree updating visit counts and value sums.

    Note: nodes already expanded (has children) are traversed during selection.
    Nodes without children are either freshly reached or terminal positions.
    """

    node = root

    # ---- Selection ----
    # Descend the tree following PUCT until we reach an unexpanded node
    while node.children:
        move, node = select_child(node)

    # ---- Expansion + Evaluation ----
    value = expand(node, model, device)

    # ---- Backpropagation ----
    backpropagate(node, value)


# =========================
# Move Selection
# =========================

def select_move(root, temperature=0):
    """
    Selects the final move to play after all simulations are complete.

    temperature=0 (greedy): always pick the most-visited move.
        Use this during actual gameplay — deterministic, picks the best move.

    temperature>0 (stochastic): sample proportional to visit_count^(1/temperature).
        Use this during self-play data generation for the first ~30 moves.
        Adds diversity so self-play games don't all follow the same path.
        temperature=1.0 samples proportional to raw visit counts.

    Args:
        root:        the root node after MCTS search
        temperature: 0 for greedy selection, >0 for stochastic sampling

    Returns:
        chess.Move to play
    """
    moves  = list(root.children.keys())
    visits = np.array([root.children[m].visits for m in moves], dtype=np.float32)

    if temperature == 0:
        return moves[np.argmax(visits)]

    # Raise visit counts to 1/temperature power, then normalise to probabilities
    visits = visits ** (1.0 / temperature)
    probs  = visits / visits.sum()
    return moves[np.random.choice(len(moves), p=probs)]


# =========================
# MCTS Search
# =========================

def mcts_search(board, model, device, simulations=1600, temperature=0):
    """
    Runs MCTS from the given board position and returns the best move.

    Process:
        1. Guard against finished positions
        2. Create root node and expand it (with Dirichlet noise)
        3. Backpropagate the initial expansion value so root.visits > 0
        4. Run N simulations (each: selection → expansion → backprop)
        5. Select move based on visit counts

    Why backpropagate the initial expansion:
        Without this, root.visits stays 0 until the first simulation completes.
        The PUCT formula uses sqrt(parent.visits + 1) which handles visits=0
        safely, but having root properly counted from the start is cleaner
        and ensures the visit distribution is accurate at the end.

    Args:
        board:       chess.Board of the current position
        model:       the neural network
        device:      torch device
        simulations: number of MCTS simulations to run (more = stronger, slower)
        temperature: 0 for greedy move selection, >0 for stochastic (self-play)

    Returns:
        chess.Move to play, or None if the game is already over
    """

    # Guard: no moves available in a finished game
    if board.is_game_over():
        return None

    root = Node(board)

    # Expand root with Dirichlet noise and backpropagate initial value
    initial_value = expand(root, model, device, add_noise=True)
    backpropagate(root, initial_value)

    # Run simulations
    for _ in range(simulations):
        run_simulation(root, model, device)

    # Select and return best move
    return select_move(root, temperature=temperature)