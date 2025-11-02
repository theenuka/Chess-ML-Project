import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

print("--- Step 7 (Fixed AGAIN): Train 'Actual Move' Model START ---")
print("WARNING: Expect very low accuracy. This is normal!")

# --- 1. Load the Processed Data ---
data_file = "processed_move_data.csv"

try:
    print(f"Loading '{data_file}'...")
    df = pd.read_csv(data_file)
    print(f"Successfully loaded {len(df)} individual moves.")
except FileNotFoundError:
    print(f"ERROR: Could not find '{data_file}'!")
    exit() # Stop the script if file is not found
except Exception as e:
    print(f"An error occurred loading the file: {e}")
    exit()

# --- 2. Define Features (X) and Target (y) ---
feature_columns = ["material_balance", "mobility", "turn"]
X = df[feature_columns]
target_column = "actual_move_played"
y = df[target_column]
unique_moves = y.nunique()

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Number of unique moves to predict (classes): {unique_moves}")

# --- 3. Split Data into Training and Testing Sets ---
#
# === THE FIX IS HERE ===
# We are reducing the sample from 30% (0.3) to 5% (0.05)
# This will use MUCH less memory.
#
print(f"Sampling 5% of the data (approx {len(X) * 0.05:.0f} moves) for this test...")
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.05, random_state=42)
# We removed 'stratify=y' in the previous step, which was correct.

print("Sampling complete. Now splitting into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# --- 4. Train Algorithm 2: Random Forest Classifier ---
print("\n--- Training Model 2: Random Forest Classifier ---")
print("This should be MUCH faster now (maybe 1-2 minutes)...")
start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=50,  # Using 50 trees
    max_depth=20,       # Limiting the depth
    random_state=42, 
    n_jobs=-1           # Use all CPU cores
)
rf_model.fit(X_train, y_train) # This is the "learning" part

end_time = time.time()
print(f"Random Forest training finished in {end_time - start_time:.2f} seconds.")

# --- 5. Test the Model ---
print("Testing the model accuracy... (This part should also work now)")
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\n--- FINAL RESULT (Move Prediction) ---")
print(f"Random Forest ACCURACY on test data: {rf_accuracy * 100:.2f}%")
print(f"\nThis means the model correctly guessed the exact move {rf_accuracy * 100:.2f}% of the time.")
print(f"This low accuracy is expected due to the {unique_moves} possible moves.")
print("--- Step 7 FINISHED ---")