import os
from dotenv import load_dotenv

import pandas as pd

def splits():
    # === CONFIGURATION ===
    load_dotenv()
    master_file = os.getenv('CSV_FILE_PATH') # Your master CSV file path
    true_output_file = os.getenv('DATA_OUTPUT')+'/true_files.txt'  # Output file for true-labeled files
    false_output_file = os.getenv('DATA_OUTPUT')+'/false_files.txt'  # Output file for other files (empty labels)

    # === STEP 1: LOAD MASTER CSV ===
    df_master = pd.read_csv(master_file)
    row_count = len(df_master)

    print(f"Number of rows (excluding header): {row_count}\n")

    # === STEP 2: FILTER FILES ===
    true_files = df_master[df_master['actualspgt20'] == True]['csvfilesortedforml'].tolist()
    false_files = df_master[df_master['actualspgt20'] != True]['csvfilesortedforml'].tolist()

    # Process files to remove the first directory name
    true_files_processed = [remove_first_directory(f) for f in true_files]
    false_files_processed = [remove_first_directory(f) for f in false_files]

    # === STEP 3: SAVE TO TEXT FILES ===

    # Write true files (with label True)
    with open(true_output_file, 'w') as f:
        for filename in true_files_processed:
            f.write(f"{filename}\n")

    # Write other files (empty labels)
    with open(false_output_file, 'w') as f:
        for filename in false_files_processed:
            f.write(f"{filename}\n")

    print(f"Created {true_output_file} with {len(true_files)} true files.")
    print(f"Created {false_output_file} with {len(false_files)} false/empty label files.")

# === Helper Function  ===
def remove_first_directory(path):
    # Remove the 'tdata/' prefix if it exists
    if path.startswith('/tdata/'):
        return path[len('/tdata/'):]
    else:
        return path
