import chess

class ChessBoard:

    def __init__(self):
        self.board = chess.Board()

    def push(self, move):
        self.board.push(move)

    def legal_moves(self):
        return list(self.board.legal_moves)

    def fen(self):
        return self.board.fen()

    def is_game_over(self):
        return self.board.is_game_over()