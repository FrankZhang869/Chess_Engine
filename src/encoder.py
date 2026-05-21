"""import chess
import numpy as np


knight_offsets = [
    (2,1),(1,2),(-1,2),(-2,1),
    (-2,-1),(-1,-2),(1,-2),(2,-1)
]

directions = [
    (0,1),(0,-1),(1,0),(-1,0),
    (1,1),(-1,1),(1,-1),(-1,-1)
]


def move_to_index(move, board):

    # Mirror move if black to move (must match mirrored board)
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

    # --------------------
    # Knight moves
    # --------------------

    for i, (r, c) in enumerate(knight_offsets):
        if (dr, dc) == (r, c):
            return from_sq * 73 + (56 + i)

    # --------------------
    # Promotions
    # --------------------

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

    # --------------------
    # Sliding moves
    # --------------------

    for i, (r, c) in enumerate(directions):

        for dist in range(1, 8):

            if (dr, dc) == (r * dist, c * dist):

                channel = i * 7 + (dist - 1)
                return from_sq * 73 + channel

    return None


def fen_to_tensor(fen):

    board = chess.Board(fen)

    # mirror board so side-to-move is always white
    if board.turn == chess.BLACK:
        board = board.mirror()

    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():

        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)

        plane = piece.piece_type - 1
        if piece.color == chess.BLACK:
            plane += 6

        tensor[plane, row, col] = 1

    return tensor"""
import chess
import numpy as np


# =========================
# Move Encoding Constants
# =========================
#
# AlphaZero encodes moves as (from_square, move_type) pairs.
# Each of the 64 squares has 73 possible move types:
#
#   Channels 0-55:  Sliding moves — 8 directions × 7 distances
#   Channels 56-63: Knight moves  — 8 possible L-shapes
#   Channels 64-72: Underpromotions — 3 piece types × 3 directions
#
# Total: 64 squares × 73 channels = 4,672 possible actions
#
# Queen promotions are NOT given a special channel — they are encoded
# as a regular sliding move to the 8th rank. This works because queen
# promotion is almost always correct and simplifies the action space.

knight_offsets = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1)
]
# Each tuple is (delta_rank, delta_file) for one of the 8 knight jumps.
# Ordering matches channel indices 56-63.

directions = [
    (0, 1), (0, -1), (1, 0), (-1, 0),       # rook directions: N S E W
    (1, 1), (-1, 1), (1, -1), (-1, -1)       # bishop directions: NE NW SE SW
]
# Each tuple is (delta_rank_per_step, delta_file_per_step).
# For each direction, distances 1-7 give channels 0-6, 7-13, ... 49-55.


# =========================
# move_to_index()
# =========================

def move_to_index(move, board):
    """
    Converts a chess.Move into an integer index in [0, 4671].

    The index encodes both WHERE the piece moves FROM and HOW it moves:
        index = from_square * 73 + move_type_channel

    Board mirroring:
        The board tensor is always mirrored so the side-to-move appears as White
        (see fen_to_tensor). Moves must be mirrored to match — a Black move on
        the real board becomes a White move on the mirrored board.

    Args:
        move:  chess.Move object (from the original, un-mirrored board)
        board: chess.Board object (used only to check whose turn it is)

    Returns:
        Integer index in [0, 4671], or None if the move cannot be encoded
        (should not happen for legal moves with a correct encoding).
    """

    # Mirror move for Black so it matches the mirrored board perspective
    if board.turn == chess.BLACK:
        move = chess.Move(
            chess.square_mirror(move.from_square),
            chess.square_mirror(move.to_square),
            promotion=move.promotion
        )

    from_sq  = move.from_square
    to_sq    = move.to_square

    from_row = chess.square_rank(from_sq)
    from_col = chess.square_file(from_sq)
    to_row   = chess.square_rank(to_sq)
    to_col   = chess.square_file(to_sq)

    dr = to_row - from_row   # rank delta
    dc = to_col - from_col   # file delta

    # -------------------------
    # Knight moves (channels 56-63)
    # -------------------------
    # Knights move in L-shapes that don't fit the sliding move pattern,
    # so they get their own 8 channels.

    for i, (r, c) in enumerate(knight_offsets):
        if (dr, dc) == (r, c):
            return from_sq * 73 + (56 + i)

    # -------------------------
    # Underpromotions (channels 64-72)
    # -------------------------
    # Only non-queen promotions need special encoding.
    # Queen promotions are handled as regular sliding moves below.
    #
    # 3 piece types: knight=0, bishop=1, rook=2
    # 3 directions:  straight=0, left=1, right=2 (capture directions)
    # → 9 channels total (64-72)

    if move.promotion is not None and move.promotion != chess.QUEEN:

        if move.promotion == chess.KNIGHT:
            piece = 0
        elif move.promotion == chess.BISHOP:
            piece = 1
        elif move.promotion == chess.ROOK:
            piece = 2
        else:
            return None     # unknown promotion piece type

        if dc == 0:
            direction = 0   # straight ahead
        elif dc < 0:
            direction = 1   # capture to the left
        else:
            direction = 2   # capture to the right

        channel = 64 + piece * 3 + direction
        return from_sq * 73 + channel

    # -------------------------
    # Sliding moves (channels 0-55)
    # -------------------------
    # Covers rooks, bishops, queens, kings (1-step), and pawns.
    # For each of 8 directions, distances 1-7 map to channels 0-6.
    # Channel = direction_index * 7 + (distance - 1)

    for i, (r, c) in enumerate(directions):
        for dist in range(1, 8):
            if (dr, dc) == (r * dist, c * dist):
                channel = i * 7 + (dist - 1)
                return from_sq * 73 + channel

    return None     # move does not match any encoding pattern


# =========================
# fen_to_tensor()
# =========================

def fen_to_tensor(fen):
    """
    Converts a FEN string into an 18-channel 8x8 numpy tensor.

    Think of it as 18 stacked 8x8 grids — one photograph of the board
    per channel, each showing a different type of information.

    Board mirroring:
        The board is always flipped so the side-to-move appears as White.
        This means the network sees the same perspective regardless of
        whose turn it is, halving the number of patterns it needs to learn.
        Mirroring flips ranks (row 0 becomes row 7) and swaps piece colors.

    Output shape: (18, 8, 8), dtype int8
    Values: 0 or 1 for piece planes; 0 or 1 for auxiliary planes

    Channel layout:
        0      White pawns
        1      White knights
        2      White bishops
        3      White rooks
        4      White queens
        5      White king
        6      Black pawns
        7      Black knights
        8      Black bishops
        9      Black rooks
        10     Black queens
        11     Black king
        12     Side to move (all 1s — always White after mirroring)
        13     White kingside castling rights (all 1s if available)
        14     White queenside castling rights (all 1s if available)
        15     Black kingside castling rights (all 1s if available)
        16     Black queenside castling rights (all 1s if available)
        17     En passant square (1 at the target square if available)

    Castling and en passant are critical:
        Without castling planes, the network cannot tell if a king can castle —
        affecting every opening and middlegame position. Without en passant,
        certain pawn captures look illegal. These 6 planes were missing from
        the original 12-plane encoder and caused significant quality loss.

    Args:
        fen: FEN string of the position to encode

    Returns:
        numpy array of shape (18, 8, 8), dtype int8
    """

    board = chess.Board(fen)

    # Mirror board so side-to-move is always White
    # After mirroring: White = the player who was originally to move
    #                  Black = the player who was originally waiting
    if board.turn == chess.BLACK:
        board = board.mirror()

    tensor = np.zeros((18, 8, 8), dtype=np.int8)

    # -------------------------
    # Planes 0-11: Piece positions
    # -------------------------
    # For each piece on the board, set the corresponding cell to 1.
    # Plane index = piece_type - 1 (pawns=0, knights=1, ..., kings=5)
    # Black pieces use planes 6-11 (add 6 to the base plane index).
    #
    # Row indexing: row 0 = rank 8 (top of board from White's view)
    #               row 7 = rank 1 (bottom of board from White's view)
    # This matches the visual board layout where White pieces start at the bottom.

    for square, piece in board.piece_map().items():
        row  = 7 - chess.square_rank(square)   # flip rank so rank 8 = row 0
        col  = chess.square_file(square)        # file a=0, b=1, ..., h=7

        plane = piece.piece_type - 1            # 0=pawn, 1=knight, ..., 5=king
        if piece.color == chess.BLACK:
            plane += 6                          # black pieces in planes 6-11

        tensor[plane, row, col] = 1

    # -------------------------
    # Plane 12: Side to move
    # -------------------------
    # Always 1 after mirroring (we are always encoding from White's perspective).
    # Kept for consistency — useful if you ever remove the mirroring convention.

    tensor[12, :, :] = 1

    # -------------------------
    # Planes 13-16: Castling rights
    # -------------------------
    # Each plane is all-1s if that castling right is available, all-0s otherwise.
    # After mirroring: White = original side to move, Black = original opponent.
    # These planes are critical for king safety evaluation.

    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1

    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1

    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1

    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1

    # -------------------------
    # Plane 17: En passant square
    # -------------------------
    # If an en passant capture is available, mark the target square with a 1.
    # En passant is only possible for one specific square at a time (or none).
    # Without this plane, certain pawn captures appear to come from nowhere,
    # confusing the network's policy head.

    if board.ep_square is not None:
        ep_row = 7 - chess.square_rank(board.ep_square)
        ep_col = chess.square_file(board.ep_square)
        tensor[17, ep_row, ep_col] = 1

    return tensor