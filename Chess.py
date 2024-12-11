# Install necessary librar
# !pip install python-chess tensorflow numpy stockfish chess

import chess
import chess.engine
import chess.pgn
import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode
from tensorflow.keras.models import Sequential, load_model
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

# Function to load games from a single PGN file
def load_games_from_file(pgn_file, limit=100000):
    positions, moves = [], []
    with open(pgn_file) as f:
        for _ in range(limit):
            game = chess.pgn.read_game(f)
            if not game:
                break

            board = chess.Board()
            for move in game.mainline_moves():
                positions.append(board_to_input(board))  # Convert board to input
                moves.append(move_to_index(move.uci()))  # Map move to index
                board.push(move)  # Apply move on board

    return np.array(positions), np.array(moves)

# Load data from multiple PGN files
def load_data_from_multiple_files(pgn_files, limit_per_file=100000):
    all_positions, all_moves = [], []
    for file in pgn_files:
        positions, moves = load_games_from_file(file, limit=limit_per_file)
        all_positions.append(positions)
        all_moves.append(moves)

    # Combine all data into single arrays
    return np.concatenate(all_positions), np.concatenate(all_moves)

# Build the CNN
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
    best_move_uci = index_to_move(best_move_index)

     # Check if the predicted move is legal before returning it
    if chess.Move.from_uci(best_move_uci) in board.legal_moves:
        return best_move_uci
    else:
        # If illegal, make a random legal move (fallback)
        return np.random.choice([m.uci() for m in board.legal_moves])

# Play a game between AI and Itself
def play_game(model):
    board = chess.Board()
    while not board.is_game_over():
        # print(board) # This prints the whole board commented because it was space consuming 
        print(f"Turn: {'White' if board.turn else 'Black'}")

        if board.turn:  # White's turn (AI)
            try:
                move = predict_best_move(board, model)
                print(f"Predicted Move: {move}")
            except Exception as e:
                print(f"Error predicting move for white: {e}")
                break
        else:  # Black's turn (random)
            try:
                move = predict_best_move(board, model)
                print(f"Predicted Move for Black: {move}")
            except Exception as e:
                print(f"Error predicting move for black: {e}")
        try:
            board.push(chess.Move.from_uci(move))
        except ValueError as e:
            print(f"Invalid move attempted: {move}, error: {e}")
            break
    print("Game Over!")
    print(f"Result: {board.result()}")


def play_against_stockfish(model, stockfish_path, elo_levels, games_per_level=10, time_limit=1.0):
    results = {elo: {"wins": 0, "losses": 0, "draws": 0, "average_cpl": []} for elo in elo_levels}

    for elo in elo_levels:
        print(f"\nTesting against Stockfish ELO {elo}...")
        for game in range(games_per_level):
            board = chess.Board()
            stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            stockfish.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})

            game_cpl = []  # Store centipawn losses for this game

            while not board.is_game_over():
                # Evaluate position before the move
                eval_before = stockfish.analyse(board, chess.engine.Limit(time=time_limit))["score"].white().score()
                
                if board.turn:  # Your model's turn (White)
                    try:
                        move = predict_best_move(board, model)
                        board.push(chess.Move.from_uci(move))
                        print(f"My models move {move}")
                    except Exception as e:
                        print(f"Error in your model's move: {e}")
                        stockfish.quit()
                        break
                else:  # Stockfish's turn (Black)
                    try:
                        result = stockfish.play(board, chess.engine.Limit(time=time_limit))
                        move = result.move
                        board.push(move)
                        print(f"Stockfish's move {move}")
                    except Exception as e:
                        print(f"Error in Stockfish's move: {e}")
                        stockfish.quit()
                        break
                print(board)        
                # Evaluate position after the move
                eval_after = stockfish.analyse(board, chess.engine.Limit(time=time_limit))["score"].white().score()

                # Calculate centipawn loss (CPL) 
                if board.turn:  # CPL For Custom Model
                    cpl = max(0, eval_before - eval_after if eval_before is not None and eval_after is not None else 0)
                    game_cpl.append(cpl)

            # Record game result
            result = board.result()
            if result == "1-0":
                results[elo]["wins"] += 1
            elif result == "0-1":
                results[elo]["losses"] += 1
            else:
                results[elo]["draws"] += 1

            # Record average CPL for the game
            if game_cpl:
                results[elo]["average_cpl"].append(np.mean(game_cpl))
            else:
                results[elo]["average_cpl"].append(None)

            print(f"Game {game + 1} against ELO {elo}: {result}")
            stockfish.quit()

    # Print summary
    print("\nSummary of Results:")
    for elo, record in results.items():
        avg_cpl = np.mean([cpl for cpl in record["average_cpl"] if cpl is not None])
        print(f"ELO {elo}: Wins: {record['wins']}, Losses: {record['losses']}, Draws: {record['draws']}, Average CPL: {avg_cpl:.2f}")


    # Print summary
    print("\nSummary of Results:")
    for elo, record in results.items():
        print(f"ELO {elo}: Wins: {record['wins']}, Losses: {record['losses']}, Draws: {record['draws']}")

# Main code
if __name__ == "__main__":
        
    # Path to the existing model
    model_path = "./chess_model.h5"

    # Load the existing model
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1)
    
    # Path to Stockfish binary
    stockfish_path = "stockfish\stockfish-windows-x86-64-avx2.exe"  

        # Define ELO levels and number of games per level
    elo_levels = [1320] #[1320, 1500, 1800]
    games_per_level = 1

    # Play games against Stockfish at different ELO levels
    play_against_stockfish(model, stockfish_path, elo_levels, games_per_level)
        
        
    # Load additional training data
    pgn_files = ["./Adams.pgn", "./Abdusattorov.pgn", "Andreikin.pgn","Carlsen.pgn","Marshall.pgn","Movsesian.pgn","Nakamura.pgn"]
    positions, move_indices = load_data_from_multiple_files(pgn_files, limit_per_file=3000)
    print(f"Combined positions shape: {positions.shape}")
    print(f"Combined move indices shape: {move_indices.shape}")
    
        ###############
        #Uncomment The below section to train the model on the data and comment the above section
        ############### 

    #     # Load the existing model
    # model_path = "./chess_model.h5"
    # try:
    #     model = load_model(model_path)
    #     print(f"Model loaded successfully from {model_path}")
    # except Exception as e:
    #     print(f"Failed to load model: {e}")
    #     exit(1)
    
    # # Compile the loaded model with a new optimizer
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # # Train the model on the combined data
    # model.fit(positions, move_indices, epochs=7, batch_size=32)

    # # Save the updated model
    # model.save(model_path)
    # print(f"Model updated and saved to {model_path}")
        
        
    
    
