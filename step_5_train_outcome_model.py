import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

print("--- Step 5: Train Game Outcome Models START ---")

# --- 1. Load the Processed Data ---
data_file = "processed_move_data.csv"

try:
    print(f"Loading '{data_file}'...")
    df = pd.read_csv(data_file)
    print(f"Successfully loaded {len(df)} individual moves.")
except FileNotFoundError:
    print(f"ERROR: Could not find '{data_file}'!")
    print("Please make sure it is in the 'Chess_Project' folder.")
    exit() # Stop the script if file is not found
except Exception as e:
    print(f"An error occurred loading the file: {e}")
    exit()

# --- 2. Define Features (X) and Target (y) ---

# Features (X): The inputs our model will use to predict
# These are the 3 features we engineered
feature_columns = ["material_balance", "mobility", "turn"]
X = df[feature_columns]

# Target (y): The output we want the model to predict
# (0 = Draw, 1 = White Win, 2 = Black Win)
target_column = "final_game_outcome"
y = df[target_column]

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# --- 3. Split Data into Training and Testing Sets ---
# We use 80% of the data to train the model, and 20% to test it.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# --- 4. Train Algorithm 1: Logistic Regression ---
# (As listed in your project proposal)
print("\n--- Training Model 1: Logistic Regression ---")
start_time = time.time()

# We set max_iter=1000 to give it enough time to find a solution
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train) # This is the "learning" part

end_time = time.time()
print(f"Logistic Regression training finished in {end_time - start_time:.2f} seconds.")

# Test the model
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)
print(f"Logistic Regression ACCURACY on test data: {lr_accuracy * 100:.2f}%")

# --- 5. Train Algorithm 2: Random Forest Classifier ---
# (As listed in your project proposal)
print("\n--- Training Model 2: Random Forest Classifier ---")
start_time = time.time()

# n_estimators=100 means it will build 100 "decision trees"
# n_jobs=-1 means it will use all available CPU cores to train faster
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train) # This is the "learning" part

end_time = time.time()
print(f"Random Forest training finished in {end_time - start_time:.2f} seconds.")

# Test the model
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest ACCURACY on test data: {rf_accuracy * 100:.2f}%")

# --- 6. Detailed Comparison Report ---
print("\n--- Detailed Classification Report (Random Forest) ---")
# This report shows more details (Precision, Recall) for each class
# We focus on Random Forest since it's likely the better model
print(classification_report(y_test, rf_predictions, target_names=["Draw (0)", "White Win (1)", "Black Win (2)"]))

print("--- Step 5 FINISHED ---")