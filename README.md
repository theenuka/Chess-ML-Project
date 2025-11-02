# EC5203 - "Pro" Chess Game Predictor (Project by Theenuka and Theekshana)

This project uses classical machine learning (Random Forest) to predict:
1.  **Game Outcome (Win/Loss/Draw %):** Based on the 65-feature board state.
2.  **Best Next Move:** Based on the 65-feature board state.

## 🚀 How to Run This "Pro" Project

You cannot run this project just by cloning it. The large data and model files are ignored by Git (`.gitignore`). You must generate them yourself by running the scripts **in the correct order.**

### Step 1: Get the Data
1.  Download the **'Chess Games'** dataset from Kaggle:
    `https://www.kaggle.com/datasets/arevel/chess-games`
2.  Download the `chess_games.csv` file (it's 4.38 GB).
3.  Place that `chess_games.csv` file inside this `Chess_Project` folder.

### Step 2: Install Libraries
You will need the following Python libraries.
```bash
pip install pandas
pip install python-chess
pip install scikit-learn
pip install matplotlib
pip install joblib