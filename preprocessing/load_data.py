import pandas as pd

def load_data(file_path):
    """Loads EEG dataset from the specified path."""
    return pd.read_csv(file_path)