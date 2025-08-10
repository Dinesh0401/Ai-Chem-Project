import os
import pandas as pd

# === CONFIG ===
RAW_DATA_DIR = os.path.join("backend", "data", "raw")
PROCESSED_DATA_DIR = os.path.join("backend", "data", "processed")
RAW_FILE_NAME = r"C:\Users\sjdin\OneDrive\Documents\Chem-Project\backend\data\raw.csv"      # Change this to your real file
PROCESSED_FILE_NAME = "processed_data.csv"

def prepare_data():
    # Ensure processed data directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    raw_file_path = os.path.join(RAW_DATA_DIR, RAW_FILE_NAME)
    processed_file_path = os.path.join(PROCESSED_DATA_DIR, PROCESSED_FILE_NAME)

    if not os.path.exists(raw_file_path):
        print(f"[ERROR] Raw data file not found at {raw_file_path}")
        return

    print(f"[INFO] Reading raw data from {raw_file_path}")
    df = pd.read_csv(raw_file_path)

    # ====== Example Cleaning Steps ======
    # Drop duplicates
    df = df.drop_duplicates()

    # Fill missing values (example: numeric columns with median)
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    # Strip whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # ====== Save Processed Data ======
    df.to_csv(processed_file_path, index=False)
    print(f"[SUCCESS] Processed data saved at {processed_file_path}")

if __name__ == "__main__":
    prepare_data()
