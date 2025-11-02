import pandas as pd

print("Python script is working...")

# Your CSV file's name
csv_file_name = "chess_games.csv" 

try:
    print(f"Attempting to read '{csv_file_name}'...")

    # Don't try to load the whole 4.38 GB file!
    # Let's use 'nrows=10' to load only the first 10 rows.
    df_head = pd.read_csv(csv_file_name, nrows=10)

    print("File was read successfully!")

    # Let's print the column names in the file
    print("\n--- Columns in the CSV File ---")
    print(list(df_head.columns))

    # Let's look at some of the data from the file
    print("\n--- First 10 Rows ---")
    print(df_head)

except FileNotFoundError:
    print(f"ERROR: Cannot find the file named '{csv_file_name}'!")
    print("Please check if the file name is correct and if it is in the 'Chess_Project' folder.")

except Exception as e:
    print(f"An unexpected ERROR occurred: {e}")