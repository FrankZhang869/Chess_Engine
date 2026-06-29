import torch
import chess
import numpy as np

from encoder import move_to_index
from encoder import fen_to_tensor


class Node:

    def __init__(self, board, parent=None, prior=0):
        self.board = board
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.visits = 0
        self.value_sum = 0

    def value(self):
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits


def select_child(node, c_puct=1.5):
    best_score = -float("inf")
    best_move = None
    best_child = None

    sqrt_parent_visits = np.sqrt(node.visits + 1)

    for move, child in node.children.items():

        Q = child.value()

        U = c_puct * child.prior * (
            sqrt_parent_visits / (1 + child.visits)
        )

        score = Q + U

        if score > best_score:
            best_score = score
            best_move = move
            best_child = child

    return best_move, best_child


def expand(node, model, device, add_noise=False):

    board = node.board

    if board.is_game_over():
        if board.is_checkmate():
            return -1
        return 0

    state = fen_to_tensor(board.fen())
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)

    with torch.no_grad():
        policy_logits, value = model(state)

    policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]

    legal_moves = list(board.legal_moves)
    valid_moves = []
    move_priors = []

    for move in legal_moves:
        idx = move_to_index(move, board)

        if idx is None:
            continue

        valid_moves.append(move)
        move_priors.append(policy[idx])

    if not valid_moves:
        return 0

    move_priors = np.array(move_priors, dtype=np.float32)

    total = np.sum(move_priors) + 1e-8
    move_priors /= total

    if add_noise and len(valid_moves) > 0:
        noise = np.random.dirichlet([0.3] * len(move_priors))
        move_priors = 0.75 * move_priors + 0.25 * noise

    for move, prior in zip(valid_moves, move_priors):
        child_board = board.copy()
        child_board.push(move)

        node.children[move] = Node(
            child_board,
            parent=node,
            prior=float(prior)
        )

    value = value.item()
    if board.turn == chess.BLACK:
        value = -value

    return value


def backpropagate(node, value):
    while node is not None:
        node.visits += 1
        node.value_sum += value
        value = -value
        node = node.parent


def run_simulation(root, model, device):

    node = root

    while node.children:
        move, node = select_child(node)

    value = expand(node, model, device)

    backpropagate(node, value)


def select_move(root, temperature=0):
    moves = list(root.children.keys())
    visits = np.array([root.children[m].visits for m in moves], dtype=np.float32)

    if temperature == 0:
        return moves[np.argmax(visits)]

    visits = visits ** (1.0 / temperature)
    probs = visits / visits.sum()
    return moves[np.random.choice(len(moves), p=probs)]


def mcts_search(board, model, device, simulations=1600, temperature=0):

    if board.is_game_over():
        return None

    root = Node(board)

    initial_value = expand(root, model, device, add_noise=True)
    backpropagate(root, initial_value)

    for _ in range(simulations):
        run_simulation(root, model, device)

    return select_move(root, temperature=temperature)
