# EC5203 - Predicting Chess Moves (Project by Theenuka and Theekshana)

This project uses classical machine learning (Random Forest, Logistic Regression) to predict chess game outcomes and optimal moves based on a large dataset of Grandmaster-level games.

## 🚀 How to Run This Project

You cannot run this project just by cloning it. The large data files are ignored by Git. You must generate them yourself.

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