import chess
import numpy as np


knight_offsets = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1)
]

directions = [
    (0, 1), (0, -1), (1, 0), (-1, 0),
    (1, 1), (-1, 1), (1, -1), (-1, -1)
]


def move_to_index(move, board):
    if board.turn == chess.BLACK:
        move = chess.Move(
            chess.square_mirror(move.from_square),
            chess.square_mirror(move.to_square),
            promotion=move.promotion
        )

    from_sq = move.from_square
    to_sq = move.to_square

    from_row = chess.square_rank(from_sq)
    from_col = chess.square_file(from_sq)
    to_row = chess.square_rank(to_sq)
    to_col = chess.square_file(to_sq)

    dr = to_row - from_row
    dc = to_col - from_col

    for i, (r, c) in enumerate(knight_offsets):
        if (dr, dc) == (r, c):
            return from_sq * 73 + (56 + i)

    if move.promotion is not None and move.promotion != chess.QUEEN:

        if move.promotion == chess.KNIGHT:
            piece = 0
        elif move.promotion == chess.BISHOP:
            piece = 1
        elif move.promotion == chess.ROOK:
            piece = 2
        else:
            return None

        if dc == 0:
            direction = 0
        elif dc < 0:
            direction = 1
        else:
            direction = 2

        channel = 64 + piece * 3 + direction
        return from_sq * 73 + channel

    for i, (r, c) in enumerate(directions):
        for dist in range(1, 8):
            if (dr, dc) == (r * dist, c * dist):
                channel = i * 7 + (dist - 1)
                return from_sq * 73 + channel

    return None


def fen_to_tensor(fen):
    board = chess.Board(fen)

    if board.turn == chess.BLACK:
        board = board.mirror()

    tensor = np.zeros((18, 8, 8), dtype=np.int8)

    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)

        plane = piece.piece_type - 1
        if piece.color == chess.BLACK:
            plane += 6

        tensor[plane, row, col] = 1

    tensor[12, :, :] = 1

    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1

    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1

    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1

    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1

    if board.ep_square is not None:
        ep_row = 7 - chess.square_rank(board.ep_square)
        ep_col = chess.square_file(board.ep_square)
        tensor[17, ep_row, ep_col] = 1

    return tensor
