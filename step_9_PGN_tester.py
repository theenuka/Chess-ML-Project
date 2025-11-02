import joblib
import chess
import chess.pgn
import io
import pandas as pd
import warnings

# Suppress a specific scikit-learn warning about feature names
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

print("--- Step 9 (FINAL DEMO - PRO/PRO Version) ---")
print("This uses the 65-feature vector for BOTH models.")

# --- 1. Helper Functions ---

def piece_to_int(piece):
    """Converts a python-chess Piece object to our 'PRO' model's integer format."""
    if piece is None: return 0  # 0 = Empty
    if piece.color == chess.WHITE: return piece.piece_type # 1-6
    else: return piece.piece_type + 6 # 7-12

def board_to_vector(board):
    """
    Converts the current board state into the 65-feature vector 
    that BOTH of our models now expect.
    """
    board_state = []
    for i in range(64):
        piece = board.piece_at(i)
        board_state.append(piece_to_int(piece))
    
    turn = 1 if board.turn == chess.WHITE else 0
    pro_feature_names = [f"sq_{i}" for i in range(64)] + ["turn"]
    pro_vector_data = board_state + [turn]
    
    # Create the final DataFrame
    input_df = pd.DataFrame([pro_vector_data], columns=pro_feature_names)
    return input_df

# --- 2. Load BOTH Saved Models ---
move_model_filename = "rf_move_model_PRO.joblib"
outcome_model_filename = "rf_outcome_model.joblib" # The file we create in step 6

try:
    print(f"Loading 'Move' model from '{move_model_filename}'...")
    move_model = joblib.load(move_model_filename)
    print(f"Loading 'Outcome' model from '{outcome_model_filename}'...")
    outcome_model = joblib.load(outcome_model_filename)
    print("All models loaded successfully!")
except FileNotFoundError as e:
    print(f"ERROR: Could not find a model file: {e}")
    print("Please run 'step_6' (PRO) and 'step_7.5' (PRO) scripts first.")
    exit()
except Exception as e:
    print(f"An error occurred loading models: {e}")
    exit()

# --- 3. Start an infinite loop to ask for PGN ---
print("\n--- Enter PGN String to Predict Move & Win % ---")
print("Example: 1. e4 e5 2. Nf3")
print("Type 'exit' to quit.")

while True:
    try:
        pgn_input = input("\nEnter PGN: ")
        if pgn_input.lower() == 'exit':
            break

        # --- 4. Parse the PGN string ---
        pgn_io = io.StringIO(pgn_input)
        game = chess.pgn.read_game(pgn_io)
        
        if game is None:
            print("Error: Invalid PGN string. Please try again.")
            continue

        board = game.end().board()
        print(f"--- Position loaded. It is {('WHITE' if board.turn else 'BLACK')}'s turn. ---")

        # --- 5. Convert Board to the ONE feature vector ---
        input_vector_df = board_to_vector(board)

        # --- 6. Make Prediction 1: Best Move ---
        print("Model (Move) is thinking...")
        move_prediction = move_model.predict(input_vector_df)
        predicted_move_uci = move_prediction[0]

        # Translate to Human-readable
        human_readable_move = ""
        try:
            move_obj = chess.Move.from_uci(predicted_move_uci)
            if move_obj in board.legal_moves:
                human_readable_move = board.san(move_obj)
            else: human_readable_move = f"{predicted_move_uci} (ILLEGAL)"
        except: human_readable_move = f"{predicted_move_uci} (INVALID)"

        # --- 7. Make Prediction 2: Win Probability ---
        print("Model (Outcome) is thinking...")
        # Send the EXACT SAME vector to the outcome model
        probabilities = outcome_model.predict_proba(input_vector_df)
        
        # Probabilities[0] looks like: [P(Draw), P(White Win), P(Black Win)]
        prob_draw = probabilities[0][0] * 100
        prob_white_win = probabilities[0][1] * 100
        prob_black_win = probabilities[0][2] * 100

        # --- 8. Show the Combined Result ---
        print("\n--- 'PRO' MODEL PREDICTIONS ---")
        print("--------------------------------------")
        print(f"  Predicted Best Move: {human_readable_move}  (UCI: {predicted_move_uci})")
        print("--------------------------------------")
        print(f"  Win Probability (based on 65 features):")
        print(f"    White Wins: {prob_white_win:.2f}%")
        print(f"    Black Wins: {prob_black_win:.2f}%")
        print(f"    Draw:       {prob_draw:.2f}%")
        print("--------------------------------------")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

print("PGN tester stopped. Goodbye!")