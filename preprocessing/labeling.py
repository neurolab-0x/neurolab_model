from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

def label_eeg_states(df: pd.DataFrame, num_clusters=3):
    """Assigns EEG states using KMeans clustering on numerical columns."""
    # Ensure there are numerical columns to process
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if numerical_cols.empty:
        raise ValueError("No numerical columns found in the dataframe.")

    # Handle edge cases with insufficient data
    if len(df) < num_clusters:
        raise ValueError(f"Number of samples ({len(df)}) is less than the number of clusters ({num_clusters}).")

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    try:
        df['eeg_state'] = kmeans.fit_predict(df[numerical_cols])
    except Exception as e:
        raise ValueError(f"Error during KMeans clustering: {e}")

    # Return dataframe with labeled states and some debug information
    print(f"KMeans clustering completed with {num_clusters} clusters.")
    print(f"Cluster centers: {kmeans.cluster_centers_}")
    print(f"Labels assigned to the dataframe rows: {df['eeg_state'].value_counts()}")
    return df
