from model.model import gan_model

import os
from dotenv import load_dotenv

import pandas as pd

def splits():
    # === CONFIGURATION ===
    load_dotenv()
    master_file = os.getenv('CSV_FILE_PATH') # Your master CSV file path
    true_output_file = os.getenv('DATA_OUTPUT')+'/true_files.csv'  # Output file for true-labeled files
    false_output_file = os.getenv('DATA_OUTPUT')+'/false_files.csv'  # Output file for other files (empty labels)

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

    # === STEP 3: SAVE TO CSV FILES ===

    pd.DataFrame({'file_path': true_files_processed}).to_csv(true_output_file, index=False)
    pd.DataFrame({'file_path': false_files_processed}).to_csv(false_output_file, index=False)

def load_data(dataset):
    if (dataset == True):
        true_files_df = pd.read_csv(os.getenv('DATA_OUTPUT')+'/true_files.csv', header=None)
        true_file_paths = true_files_df[0].tolist()

        for file_path in true_file_paths:
            try:
                gan_model(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    else:
        false_files_df = pd.read_csv(os.getenv('DATA_OUTPUT')+'/false_files.csv', header=None)
        false_file_paths = false_files_df[0].tolist()
        
        for file_path in false_file_paths:
            try:
                gan_model(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


# === Helper Function  ===
def remove_first_directory(path):
    # Remove the 'tdata/' prefix if it exists
    if path.startswith('/tdata/'):
        return path[len('/tdata/'):]
    else:
        return path
