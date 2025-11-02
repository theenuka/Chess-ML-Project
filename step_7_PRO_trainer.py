import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # <-- NEW: Import joblib to save the model
import time

print("--- Step 7.5 (PRO): Train & SAVE Move Model START ---")

# --- 1. Load the PROCESSED Data ---
data_file = "processed_move_data_PRO.csv" # The NEW file

try:
    print(f"Loading '{data_file}'...")
    df = pd.read_csv(data_file)
    print(f"Successfully loaded {len(df)} individual moves.")
except FileNotFoundError:
    print(f"ERROR: Could not find '{data_file}'!")
    exit()
except Exception as e:
    print(f"An error occurred loading the file: {e}")
    exit()

# --- 2. Define Features (X) and Target (y) ---
feature_columns = [f"sq_{i}" for i in range(64)] + ["turn"]
X = df[feature_columns]
y = df["actual_move_played"]
unique_moves = y.nunique()

print(f"Features (X) shape: {X.shape} (65 features!)")
print(f"Number of unique moves to predict (classes): {unique_moves}")

# --- 3. Split Data into Training and Testing Sets ---
SAMPLE_SIZE = 0.05 # Using 5% sample
print(f"Sampling {SAMPLE_SIZE * 100}% of the data...")
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=SAMPLE_SIZE, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# --- 4. Train Random Forest Classifier ---
print("\n--- Training Model 2: Random Forest Classifier ---")
start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=50,  
    max_depth=30,       
    random_state=42, 
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

end_time = time.time()
print(f"Random Forest training finished in {end_time - start_time:.2f} seconds.")

# --- 5. Test the Model (for our confirmation) ---
print("Testing the model accuracy...")
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Model Accuracy (for confirmation): {rf_accuracy * 100:.2f}%")


# --- 6. NEW: Save the Trained Model to a File ---
model_filename = "rf_move_model_PRO.joblib"
print(f"\nSaving the trained 'PRO' move model to '{model_filename}'...")
joblib.dump(rf_model, model_filename)
print("Model saved successfully!")

print(f"\n--- Step 7.5 (PRO) FINISHED ---")