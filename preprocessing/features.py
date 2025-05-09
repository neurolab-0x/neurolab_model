from scipy.signal import welch, butter, filtfilt
import numpy as np
from scipy.stats import skew, kurtosis, entropy
import pandas as pd
from scipy.integrate import simpson
import antropy as ant

def compute_psd(signal, fs=250):
    """Computes PSD while ensuring nperseg is not greater than signal length."""
    nperseg = min(256, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    return freqs, psd

def compute_band_power(freqs, psd, band):
    """Calculate power in specific frequency band."""
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    if not any(idx):  # If no frequencies in the band
        return 0
    return simpson(psd[idx], freqs[idx])

def compute_hjorth_parameters(data):
    """Computes Hjorth parameters: Activity, Mobility, and Complexity."""
    # Activity - variance of the signal
    activity = np.var(data)
    
    # First derivative
    diff1 = np.diff(data)
    # Mobility
    mobility = np.sqrt(np.var(diff1) / activity) if activity > 0 else 0
    
    # Second derivative
    diff2 = np.diff(diff1)
    # Complexity
    complexity = np.sqrt(np.var(diff2) / np.var(diff1)) / mobility if mobility > 0 and np.var(diff1) > 0 else 0
    
    return activity, mobility, complexity

def compute_spectral_entropy(psd):
    """Compute normalized spectral entropy of the signal."""
    psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
    return entropy(psd_norm)

def extract_features(df):
    """Extracts advanced statistical, spectral, and non-linear features."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    feature_list = []
    
    # Define standard EEG frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }

    for i, row in df.iterrows():
        feature_data = {}
        
        # Process each channel
        for col in numerical_cols:
            if col != 'eeg_state':
                # Get signal data
                signal = np.array([row[col]]) if isinstance(row[col], (int, float)) else row[col]
                
                # Basic statistical features
                feature_data[f'{col}_mean'] = np.mean(signal)
                feature_data[f'{col}_std'] = np.std(signal)
                feature_data[f'{col}_var'] = np.var(signal)
                feature_data[f'{col}_skew'] = skew(signal) if len(signal) > 2 else 0
                feature_data[f'{col}_kurtosis'] = kurtosis(signal) if len(signal) > 2 else 0
                
                # Time domain features
                feature_data[f'{col}_zero_crossings'] = np.sum(np.diff(np.signbit(signal).astype(int)) != 0)
                
                # Hjorth parameters
                activity, mobility, complexity = compute_hjorth_parameters(signal)
                feature_data[f'{col}_activity'] = activity
                feature_data[f'{col}_mobility'] = mobility
                feature_data[f'{col}_complexity'] = complexity
                
                # Frequency domain features
                try:
                    freqs, psd = compute_psd(signal)
                    
                    # Band powers
                    for band_name, band_range in bands.items():
                        band_power = compute_band_power(freqs, psd, band_range)
                        feature_data[f'{col}_{band_name}_power'] = band_power
                    
                    # Band power ratios (useful for mental state classification)
                    alpha_power = compute_band_power(freqs, psd, bands['alpha'])
                    beta_power = compute_band_power(freqs, psd, bands['beta'])
                    theta_power = compute_band_power(freqs, psd, bands['theta'])
                    
                    # Calculate ratios (avoid division by zero)
                    if beta_power > 0:
                        feature_data[f'{col}_alpha_beta_ratio'] = alpha_power / beta_power
                    else:
                        feature_data[f'{col}_alpha_beta_ratio'] = 0
                        
                    if theta_power > 0:
                        feature_data[f'{col}_beta_theta_ratio'] = beta_power / theta_power
                    else:
                        feature_data[f'{col}_beta_theta_ratio'] = 0
                    
                    # Spectral entropy
                    feature_data[f'{col}_spectral_entropy'] = compute_spectral_entropy(psd)
                    
                except Exception:
                    # Fill with zeros if computation fails
                    for band_name in bands:
                        feature_data[f'{col}_{band_name}_power'] = 0
                    feature_data[f'{col}_alpha_beta_ratio'] = 0
                    feature_data[f'{col}_beta_theta_ratio'] = 0
                    feature_data[f'{col}_spectral_entropy'] = 0
                
                # Non-linear features (if signal is long enough)
                if len(signal) > 10:
                    try:
                        # Sample entropy
                        feature_data[f'{col}_sample_entropy'] = ant.sample_entropy(signal)
                        # Approximate entropy
                        feature_data[f'{col}_app_entropy'] = ant.app_entropy(signal)
                        # Permutation entropy
                        feature_data[f'{col}_perm_entropy'] = ant.perm_entropy(signal)
                    except Exception:
                        feature_data[f'{col}_sample_entropy'] = 0
                        feature_data[f'{col}_app_entropy'] = 0
                        feature_data[f'{col}_perm_entropy'] = 0
                else:
                    feature_data[f'{col}_sample_entropy'] = 0
                    feature_data[f'{col}_app_entropy'] = 0
                    feature_data[f'{col}_perm_entropy'] = 0
        
        # Multi-channel features if multiple EEG channels exist
        eeg_channels = [col for col in numerical_cols if col != 'eeg_state']
        if len(eeg_channels) > 1:
            # Compute basic cross-channel correlations
            for i, ch1 in enumerate(eeg_channels[:-1]):
                for ch2 in eeg_channels[i+1:]:
                    sig1 = np.array([row[ch1]]) if isinstance(row[ch1], (int, float)) else row[ch1]
                    sig2 = np.array([row[ch2]]) if isinstance(row[ch2], (int, float)) else row[ch2]
                    
                    if len(sig1) > 1 and len(sig2) > 1:
                        # Compute correlation
                        try:
                            corr = np.corrcoef(sig1, sig2)[0, 1]
                            feature_data[f'{ch1}_{ch2}_corr'] = corr if not np.isnan(corr) else 0
                        except Exception:
                            feature_data[f'{ch1}_{ch2}_corr'] = 0
                    else:
                        feature_data[f'{ch1}_{ch2}_corr'] = 0
        
        # Keep original label
        if 'eeg_state' in row:
            feature_data['eeg_state'] = row['eeg_state']
            
        feature_list.append(feature_data)

    return pd.DataFrame(feature_list)
