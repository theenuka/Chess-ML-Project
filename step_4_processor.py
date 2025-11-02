import pandas as pd
import chess
import chess.pgn
import io  # Required to read a string as if it were a file
import time # To time how long the script takes

print("--- Step 4: Top Player Data Processor START ---")

# --- 1. Helper Functions (English Comments) ---

def piece_value(piece_type):
    """Gets the integer value for a chess piece."""
    if piece_type == chess.PAWN: return 1
    if piece_type == chess.KNIGHT: return 3
    if piece_type == chess.BISHOP: return 3
    if piece_type == chess.ROOK: return 5
    if piece_type == chess.QUEEN: return 9
    return 0 # King value is not needed for material count

def get_outcome_numeric(result_string):
    """Converts the result string ('1-0') to a number."""
    if result_string == "1-0":
        return 1  # White win
    elif result_string == "0-1":
        return 2  # Black win
    else:
        return 0  # Draw (e.g., '1/2-1/2')

def process_game_row(game_row):
    """
    Takes a single game row from our DataFrame,
    parses the 'AN' string, and extracts features for *every* move.
    """
    
    # 1. Get game details from the row
    game_an = game_row['AN']
    game_result = game_row['Result']
    final_outcome_numeric = get_outcome_numeric(game_result)

    # 2. Convert 'AN' string to a PGN format that python-chess can read
    # We must add a header (like [Result]) for it to be valid PGN.
    pgn_string = f"[Result \"{game_result}\"]\n{game_an}"
    pgn_io = io.StringIO(pgn_string)
    
    try:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            return [] # Return an empty list if game is unreadable
    except Exception as e:
        # print(f"Warning: Skipping a game due to parsing error: {e}")
        return [] # Skip this game if there's an error

    # 3. Now, loop through every move in this one game
    board = game.board()
    features_for_this_game = [] # A list to store all move-data from this game

    for move in game.mainline_moves():
        
        # --- Feature Engineering (As per your proposal) ---
        
        # Feature 1: Material Balance (from White's perspective)
        material_balance = 0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            material_balance += len(board.pieces(piece_type, chess.WHITE)) * piece_value(piece_type)
            material_balance -= len(board.pieces(piece_type, chess.BLACK)) * piece_value(piece_type)

        # Feature 2: Mobility (number of legal moves for the current player)
        mobility = board.legal_moves.count()

        # Feature 3: Turn (White = 1, Black = 0)
        turn = 1 if board.turn == chess.WHITE else 0

        # --- Target Variables (What we want to predict) ---
        
        # Target 1: The Move (The actual move that was played)
        actual_move = move.uci() # e.g., "e2e4"

        # Target 2: Game Outcome (Who won at the *end* of the game)
        # We already calculated this as 'final_outcome_numeric'
        
        # --- Collate all data for this single move ---
        current_move_data = {
            "material_balance": material_balance,
            "mobility": mobility,
            "turn": turn,
            "actual_move_played": actual_move,
            "final_game_outcome": final_outcome_numeric
        }
        features_for_this_game.append(current_move_data)
        
        # --- Push the move to the board to prepare for the next loop ---
        board.push(move)
    
    return features_for_this_game

# --- 2. Main Script Logic ---

input_csv = "chess_games_TOP_PLAYERS.csv"
output_csv = "processed_move_data.csv"

print(f"Loading '{input_csv}'... This is our high-quality dataset.")
start_time = time.time() # Let's time this

try:
    # Read the entire TOP_PLAYERS file. 4198 rows is small, no chunks needed.
    df = pd.read_csv(input_csv)
    
    print(f"Successfully loaded {len(df)} games.")
    print("Now processing every move in every game... This might take 1-2 minutes.")
    
    all_moves_data = [] # A master list to hold ALL moves from ALL games

    # Loop through each game (row) in our DataFrame
    for index, row in df.iterrows():
        # Process the game
        move_data = process_game_row(row)
        
        # Add the list of moves from that game to our master list
        all_moves_data.extend(move_data)
        
        if (index + 1) % 500 == 0: # Print a progress update every 500 games
            print(f"  ...processed {index + 1} / {len(df)} games.")

    print(f"All {len(df)} games processed!")
    print(f"Total individual moves extracted: {len(all_moves_data)}")

    # Convert the master list of moves into a final DataFrame
    final_df = pd.DataFrame(all_moves_data)
    
    # Save the final DataFrame to our new CSV
    final_df.to_csv(output_csv, index=False)
    
    end_time = time.time()
    
    print("\n--- SUCCESS! ---")
    print(f"Successfully created '{output_csv}'.")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")

except FileNotFoundError:
    print(f"ERROR: Cannot find the file '{input_csv}'")
    print("Please make sure it is in the same 'Chess_Project' folder.")
except Exception as e:
    print(f"\n--- ERROR! ---")
    print(f"An unexpected error occurred: {e}")