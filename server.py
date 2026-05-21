from flask import Flask, request, jsonify, send_file
import chess
import torch

from model import ChessNet2
from mcts import mcts_search

app = Flask(__name__)

# =========================
# Load model
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ChessNet2().to(device)
model.load_state_dict(torch.load("/Users/frankzhang/Downloads/model_iter_5.pt", map_location=device))
model.eval()

# After loading model, before starting the game:
def warmup_model(model, device):
    print("Warming up model...")
    dummy = torch.zeros(1, 18, 8, 8).to(device)
    with torch.no_grad():
        for _ in range(3):   # 3 passes ensures all caches are warm
            model(dummy)
    print("Ready.")

warmup_model(model, device)

# =========================
# Game board
# =========================

board = chess.Board()

# =========================
# Serve the webpage
# =========================

@app.route("/")
def home():
    return send_file("index.html")


# =========================
# Handle player move
# =========================

@app.route("/move", methods=["POST"])
def move():

    data = request.json
    user_move = data["move"]

    try:
        board.push_uci(user_move)
    except:
        return jsonify({"error": "illegal move"})

    # Engine move
    engine_move = mcts_search(board, model, device, simulations=400)

    board.push(engine_move)

    return jsonify({"move": engine_move.uci()})


# =========================
# Reset board (optional)
# =========================

@app.route("/reset")
def reset():

    global board
    board = chess.Board()

    return jsonify({"status": "reset"})


# =========================
# Run server
# =========================

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)