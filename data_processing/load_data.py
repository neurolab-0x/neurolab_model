import os
import pandas as pd
from data_processing.preprocess import remove_outliers, scale_features, add_mental_state_labels
import numpy as np
from sklearn.impute import SimpleImputer

def load_and_clean_data(file_path):
  dataframe = pd.read_csv(file_path)
  dataframe.dropna(inplace=True)
  return dataframe

def load_multiple_datasets(directory):
    """
    Load and preprocess multiple datasets from the given directory.

    Args:
        directory (str): Path to the directory containing datasets.

    Returns:
        pd.DataFrame: Combined and preprocessed DataFrame.
    """
    combined_df = pd.DataFrame()
    reference_columns = None
    missing_columns_summary = {}
    extra_columns_summary = {}

    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if file.endswith('.csv'):
            print(f"Processing dataset: {file}")
            df = pd.read_csv(file_path)
            df.dropna(inplace=True)  # Drop rows with missing values

            # Check column consistency
            if reference_columns is None:
                reference_columns = df.columns
            else:
                # Handle missing and extra columns
                missing_cols = set(reference_columns) - set(df.columns)
                extra_cols = set(df.columns) - set(reference_columns)

                if missing_cols:
                    missing_columns_summary[file] = missing_cols
                    for col in missing_cols:
                        df[col] = np.nan  # Add missing columns with NaN

                if extra_cols:
                    extra_columns_summary[file] = extra_cols
                    df = df.drop(columns=extra_cols, errors='ignore')

            # Ensure column order matches reference
            df = df[reference_columns]

            # Preprocessing
            df = remove_outliers(df)
            if df.empty:
                print(f"Dataset {file} is empty after outlier removal. Skipping...")
                continue

            df = scale_features(df)
            df = add_mental_state_labels(df)

            try:
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            except MemoryError:
                print(f"MemoryError: Unable to combine dataset {file}. Skipping...")

    # Check combined dataset
    if combined_df.empty:
        raise ValueError("Combined dataset is empty after processing all files.")

    print("All datasets loaded and combined.")
    print(f"Combined dataset shape: {combined_df.shape}")

    if missing_columns_summary:
        print("Summary of Missing Columns:")
        for file, cols in missing_columns_summary.items():
            print(f"  {file}: {cols}")

    if extra_columns_summary:
        print("Summary of Extra Columns:")
        for file, cols in extra_columns_summary.items():
            print(f"  {file}: {cols}")

    # Handle missing values in numeric columns
    numeric_columns = combined_df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='mean')
    combined_df[numeric_columns] = imputer.fit_transform(combined_df[numeric_columns])

    return combined_df
