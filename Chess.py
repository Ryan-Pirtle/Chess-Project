# Install necessary libraries
# pip install python-chess tensorflow numpy

import chess
import chess.pgn
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D

# Mapping UCI moves to indices and back
def move_to_index(move):
    files = "abcdefgh"
    ranks = "12345678"
    start_file = files.index(move[0])
    start_rank = ranks.index(move[1])
    end_file = files.index(move[2])
    end_rank = ranks.index(move[3])
    return (start_rank * 8 + start_file) * 64 + (end_rank * 8 + end_file)

def index_to_move(index):
    files = "abcdefgh"
    ranks = "12345678"
    start = divmod(index // 64, 8)
    end = divmod(index % 64, 8)
    return f"{files[start[1]]}{ranks[start[0]]}{files[end[1]]}{ranks[end[0]]}"

# Generate legal move mask
def generate_legal_move_mask(board):
    mask = np.zeros(4096)
    for move in board.legal_moves:
        index = move_to_index(move.uci())
        mask[index] = 1
    return mask

# Normalize predictions to a probability distribution
def normalize_predictions(masked_predictions):
    total = np.sum(masked_predictions)
    if total > 0:
        return masked_predictions / total
    else:
        raise ValueError("No valid moves available in the predictions!")

# Convert chess board to a one-hot encoded array
def board_to_input(board):
    piece_map = {
        'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,
        'P': 7, 'N': 8, 'B': 9, 'R': 10, 'Q': 11, 'K': 12
    }
    board_array = np.zeros((8, 8, 12))
    for square, piece in board.piece_map().items():
        x, y = divmod(square, 8)
        channel = piece_map[piece.symbol()]
        board_array[x, y, channel - 1] = 1
    return board_array

# Load chess games from PGN file
def load_games(pgn_file, limit=1000):
    positions, moves = [], []
    with open(pgn_file) as f:
        for _ in range(limit):
            game = chess.pgn.read_game(f)
            if not game:
                break

            board = chess.Board()
            for move in game.mainline_moves():
                positions.append(board_to_input(board))
                moves.append(move.uci())
                board.push(move)

    return np.array(positions), np.array([move_to_index(move) for move in moves])

# Build the neural network model
def build_model():
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(8, 8, 12)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(4096, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Predict the best move for a given board state
def predict_best_move(board, model):
    input_data = np.expand_dims(board_to_input(board), axis=0)
    predictions = model.predict(input_data)[0]

    legal_move_mask = generate_legal_move_mask(board)
    masked_predictions = predictions * legal_move_mask
    normalized_predictions = normalize_predictions(masked_predictions)

    best_move_index = np.argmax(normalized_predictions)
    best_move = index_to_move(best_move_index)

    if best_move not in [m.uci() for m in board.legal_moves]:
        raise ValueError("Predicted an illegal move.")

    return best_move

# Play a game between AI and a random player
def play_game(model):
    board = chess.Board()
    while not board.is_game_over():
        print(board)
        print(f"Turn: {'White' if board.turn else 'Black'}")
        
        if board.turn:  # White's turn (AI)
            try:
                move = predict_best_move(board, model)
                print(f"Predicted Move: {move}")
            except Exception as e:
                print(f"Error predicting move: {e}")
                break
        else:  # Black's turn (random)
            move = np.random.choice([m.uci() for m in board.legal_moves])
            print(f"Random Move for Black: {move}")

        try:
            board.push(chess.Move.from_uci(move))
        except ValueError as e:
            print(f"Invalid move attempted: {move}, error: {e}")
            break

    print("Game Over!")
    print(f"Result: {board.result()}")

# Main code
if __name__ == "__main__":
    # Load data
    pgn_file = "./Abdusattorov.pgn"  # Replace with your PGN file
    positions, move_indices = load_games(pgn_file, limit=1000)

    # Build and train the model
    model = build_model()
    model.fit(positions, move_indices, epochs=10, batch_size=32)

    # Save the model
    model.save("chess_model.h5")

    # Play a game
    play_game(model)
