import pandas as pd
import chess
import chess.pgn
import io
import time

print("--- Step 4 (PRO): Board-State Processor START ---")
print("This will create a MUCH larger data file.")

# --- 1. Helper Functions ---

def piece_to_int(piece):
    """Converts a python-chess Piece object to a simple integer."""
    if piece is None:
        return 0  # 0 = Empty
    
    # Positive for White, Negative for Black
    # We will use 1-6 for White, 7-12 for Black
    # 1:P, 2:N, 3:B, 4:R, 5:Q, 6:K
    # 7:p, 8:n, 9:b, 10:r, 11:q, 12:k
    
    if piece.color == chess.WHITE:
        return piece.piece_type # 1 (Pawn) to 6 (King)
    else:
        # For Black, add 6
        return piece.piece_type + 6 # 7 (Pawn) to 12 (King)

def get_outcome_numeric(result_string):
    """Converts '1-0' to 1 (White Win), '0-1' to 2 (Black Win), else 0 (Draw)."""
    if result_string == "1-0": return 1
    elif result_string == "0-1": return 2
    else: return 0

def process_game_row(game_row):
    """
    Takes a single game row, parses the 'AN' string,
    and extracts a 64-feature board vector for *every* move.
    """
    game_an = game_row['AN']
    game_result = game_row['Result']
    final_outcome_numeric = get_outcome_numeric(game_result)

    pgn_string = f"[Result \"{game_result}\"]\n{game_an}"
    pgn_io = io.StringIO(pgn_string)
    
    try:
        game = chess.pgn.read_game(pgn_io)
        if game is None: return []
    except Exception as e:
        return []

    board = game.board()
    features_for_this_game = []

    for move in game.mainline_moves():
        
        # --- NEW Feature Engineering: Board State Vector (64 features) ---
        board_state = []
        # We loop through all 64 squares (from a1 to h8)
        for i in range(64):
            piece = board.piece_at(i)
            board_state.append(piece_to_int(piece))
        
        # Feature 65: Turn (1 for White, 0 for Black)
        turn = 1 if board.turn == chess.WHITE else 0
        
        # Target 1: The Move
        actual_move = move.uci() # e.g., "e2e4"
        
        # Target 2: Game Outcome
        # (We keep this for our *other* model, just in case)
        
        # Create the data row
        # We start with the 64 board features
        current_move_data = {f"sq_{i}": val for i, val in enumerate(board_state)}
        
        # Add the other features
        current_move_data["turn"] = turn
        current_move_data["actual_move_played"] = actual_move
        current_move_data["final_game_outcome"] = final_outcome_numeric
        
        features_for_this_game.append(current_move_data)
        
        board.push(move)
    
    return features_for_this_game

# --- 2. Main Script Logic ---
input_csv = "chess_games_TOP_PLAYERS.csv"
output_csv = "processed_move_data_PRO.csv" # New output file

print(f"Loading '{input_csv}'...")
start_time = time.time()

try:
    df = pd.read_csv(input_csv)
    print(f"Successfully loaded {len(df)} games.")
    print("Now processing every move... This might take 1-2 minutes.")
    
    all_moves_data = [] 

    for index, row in df.iterrows():
        move_data = process_game_row(row)
        all_moves_data.extend(move_data)
        
        if (index + 1) % 500 == 0:
            print(f"  ...processed {index + 1} / {len(df)} games.")

    print(f"All {len(df)} games processed!")
    print(f"Total individual moves extracted: {len(all_moves_data)}")

    final_df = pd.DataFrame(all_moves_data)
    
    # This file will be MUCH larger. Let's save it.
    print(f"Saving to '{output_csv}'...")
    final_df.to_csv(output_csv, index=False)
    
    end_time = time.time()
    
    print("\n--- SUCCESS! (PRO) ---")
    print(f"Successfully created '{output_csv}'.")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")

except FileNotFoundError:
    print(f"ERROR: Cannot find the file '{input_csv}'")
except Exception as e:
    print(f"\n--- ERROR! ---: {e}")