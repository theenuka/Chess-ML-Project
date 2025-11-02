import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt # For plotting
import time

print("--- Step 6: Model Explainability (Feature Importance) START ---")

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
target_column = "final_game_outcome"

X = df[feature_columns]
y = df[target_column]

print("Features (X) and Target (y) are ready.")

# We don't need a train/test split here, 
# as we want to train on ALL data to find the *overall* feature importance.
# This is a common practice for feature importance analysis.

# --- 3. Train the Random Forest Model (on ALL data) ---
print("\n--- Training Random Forest on ALL data ---")
print("(This will take about 20-30 seconds...)")
start_time = time.time()

# We use the same settings as before
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X, y) # This is the "learning" part, on all data

end_time = time.time()
print(f"Random Forest training finished in {end_time - start_time:.2f} seconds.")

# --- 4. Get Feature Importances ---
print("Extracting feature importances from the model...")
importances = rf_model.feature_importances_ 
# This gives a list of scores, e.g., [0.8, 0.15, 0.05]

# --- 5. Create and Save the Bar Chart ---
output_image_file = "feature_importance_chart.png"

print(f"Creating bar chart and saving to '{output_image_file}'...")

plt.figure(figsize=(10, 6)) # Set the figure size
plt.bar(feature_columns, importances, color=['blue', 'green', 'red'])
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.title("Which Feature is Most Important for Predicting Game Outcome?")
plt.tight_layout() # Make sure labels fit

# Save the plot to a file
plt.savefig(output_image_file) 

print("\n--- SUCCESS! ---")
print(f"Successfully created '{output_image_file}'.")
print("Go to your 'Chess_Project' folder and open the image!")

print("\n--- Importance Scores ---")
for feature, score in zip(feature_columns, importances):
    print(f"Feature: {feature}, Importance: {score * 100:.2f}%")

print("--- Step 6 FINISHED ---")