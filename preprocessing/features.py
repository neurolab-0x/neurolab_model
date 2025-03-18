from scipy.signal import welch
import numpy as np
from scipy.stats import skew, kurtosis
import pandas as pd

def compute_psd(signal, fs=250):
    """Computes PSD while ensuring nperseg is not greater than signal length."""
    nperseg = min(256, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    return freqs, psd

def extract_features(df):
    """Extracts statistical and frequency-based features row-wise."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    feature_list = []

    for i, row in df.iterrows():
        feature_data = {}
        for col in numerical_cols:
            if col != 'eeg_state':
                feature_data[f'{col}_mean'] = row[col]
                feature_data[f'{col}_variance'] = row[col] ** 2
                feature_data[f'{col}_skew'] = 0 if row[col] == 0 else row[col] / abs(row[col])
                feature_data[f'{col}_kurtosis'] = 3
                
                freqs, psd = compute_psd(np.array([row[col]]))
                feature_data[f'{col}_alpha'] = np.mean(psd[(freqs >= 8) & (freqs <= 12)])
                feature_data[f'{col}_beta'] = np.mean(psd[(freqs >= 12) & (freqs <= 30)])

        feature_data['eeg_state'] = row['eeg_state']
        feature_list.append(feature_data)

    return pd.DataFrame(feature_list)
