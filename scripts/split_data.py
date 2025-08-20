import pandas as pd

SOURCE_DATA_PATH = "binance_data_2023-01-01_to_2025-08-18.csv"
TRAINING_OUTPUT_PATH = "training_data.csv"
TESTING_OUTPUT_PATH = "testing_data.csv"
TEST_DURATION_DAYS = 30 # dados de 30 dias p testar

def split_dataset():
    print(f"--- Starting Dataset Split ---")
    try:
        print(f"Loading source file: {SOURCE_DATA_PATH}")
        df = pd.read_csv(SOURCE_DATA_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except FileNotFoundError:
        print(f"ERROR: Source file '{SOURCE_DATA_PATH}' not found.")
        return
    last_date = df['timestamp'].max()
    split_date = last_date - pd.Timedelta(days=TEST_DURATION_DAYS)
    print(f"Split date: {split_date.date()}")
    training_df = df[df['timestamp'] < split_date]
    testing_df = df[df['timestamp'] >= split_date]
    if training_df.empty or testing_df.empty:
        print("ERROR: Failed to split dataset.")
        return
    print(f"ℹSaving training file with {len(training_df)} rows to '{TRAINING_OUTPUT_PATH}'...")
    training_df.to_csv(TRAINING_OUTPUT_PATH, index=False)
    print(f"ℹSaving testing file with {len(testing_df)} rows to '{TESTING_OUTPUT_PATH}'...")
    testing_df.to_csv(TESTING_OUTPUT_PATH, index=False)
    print("\nSplit complete!")

if __name__ == "__main__":
    split_dataset()