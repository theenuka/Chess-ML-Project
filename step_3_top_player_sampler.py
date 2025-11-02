import pandas as pd

print("--- New Step 3: Top Player Sampler START ---")

input_csv = "chess_games.csv"
output_sample_csv = "chess_games_TOP_PLAYERS.csv"

# Let's set a rating threshold. 2500+ is Grandmaster level.
# You can lower this to 2400 if you want.
RATING_THRESHOLD = 2500 

chunk_size = 50000  # Let's read 50,000 rows at a time
is_first_chunk = True

print(f"Reading '{input_csv}' file in chunks...")
print(f"Saving games with Rating > {RATING_THRESHOLD} to '{output_sample_csv}'...")
print("This might also take a while (1-2 minutes)...")

try:
    total_games_found = 0

    # Read the large file in pieces (chunks)
    for chunk in pd.read_csv(input_csv, chunksize=chunk_size):

        # Convert rating columns to numbers ('int').
        # If there is data like '?', 'errors='coerce'' will skip it (by making it NaN).
        chunk['WhiteElo'] = pd.to_numeric(chunk['WhiteElo'], errors='coerce')
        chunk['BlackElo'] = pd.to_numeric(chunk['BlackElo'], errors='coerce')

        # Remove rows where the rating became NaN (empty/invalid)
        chunk = chunk.dropna(subset=['WhiteElo', 'BlackElo'])

        # --- Here is the Filter ---
        # Both White's AND Black's rating must be greater than the threshold
        top_games_chunk = chunk[
            (chunk['WhiteElo'] > RATING_THRESHOLD) & 
            (chunk['BlackElo'] > RATING_THRESHOLD)
        ]

        # If any top-rated games were found...
        if not top_games_chunk.empty:
            total_games_found += len(top_games_chunk)

            # Save them to the file
            if is_first_chunk:
                # The first chunk is saved with the header (column names)
                top_games_chunk.to_csv(output_sample_csv, mode='w', header=True, index=False)
                is_first_chunk = False
            else:
                # Other chunks are appended to the end without the header
                top_games_chunk.to_csv(output_sample_csv, mode='a', header=False, index=False)

            print(f"Found {len(top_games_chunk)} games with Rating > {RATING_THRESHOLD}... (Current Total: {total_games_found})")

    print(f"\n--- Success! ---")
    print(f"Created the file '{output_sample_csv}'.")
    print(f"Found a total of {total_games_found} games with Rating > {RATING_THRESHOLD}.")

except Exception as e:
    print(f"An ERROR occurred: {e}")