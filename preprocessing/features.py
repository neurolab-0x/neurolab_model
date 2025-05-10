from scipy.signal import welch, butter, filtfilt, hilbert, coherence
import numpy as np
from scipy.stats import skew, kurtosis, entropy, pearsonr
import pandas as pd
from scipy.integrate import simpson
import antropy as ant
import logging
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import warnings
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import pywt
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

class FeatureExtractionError(Exception):
    """Custom exception for feature extraction errors"""
    pass

def compute_psd(signal: np.ndarray, fs: float = 250) -> Tuple[np.ndarray, np.ndarray]:
    """Enhanced PSD computation with validation"""
    try:
        if len(signal) < 2:
            raise FeatureExtractionError("Signal too short for PSD computation")
        
        nperseg = min(256, len(signal))
        freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
        return freqs, psd
    except Exception as e:
        raise FeatureExtractionError(f"Error in PSD computation: {str(e)}")

def compute_band_power(freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> float:
    """Enhanced band power computation with validation"""
    try:
        idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        if not any(idx):
            return 0
        return simpson(psd[idx], freqs[idx])
    except Exception as e:
        raise FeatureExtractionError(f"Error in band power computation: {str(e)}")

def compute_hjorth_parameters(data: np.ndarray) -> Tuple[float, float, float]:
    """Enhanced Hjorth parameters computation with validation"""
    try:
        if len(data) < 2:
            return 0, 0, 0
        
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
    except Exception as e:
        raise FeatureExtractionError(f"Error in Hjorth parameters computation: {str(e)}")

def compute_spectral_entropy(psd: np.ndarray) -> float:
    """Enhanced spectral entropy computation with validation"""
    try:
        psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
        return entropy(psd_norm)
    except Exception as e:
        raise FeatureExtractionError(f"Error in spectral entropy computation: {str(e)}")

def compute_nonlinear_features(signal: np.ndarray) -> Dict[str, float]:
    """Compute nonlinear features with error handling"""
    features = {}
    try:
        if len(signal) > 10:
            features['sample_entropy'] = ant.sample_entropy(signal)
            features['app_entropy'] = ant.app_entropy(signal)
            features['perm_entropy'] = ant.perm_entropy(signal)
            features['spectral_entropy'] = ant.spectral_entropy(signal, sf=250)
            features['svd_entropy'] = ant.svd_entropy(signal)
        else:
            features.update({k: 0 for k in ['sample_entropy', 'app_entropy', 'perm_entropy', 
                                          'spectral_entropy', 'svd_entropy']})
    except Exception as e:
        logger.warning(f"Error computing nonlinear features: {str(e)}")
        features.update({k: 0 for k in ['sample_entropy', 'app_entropy', 'perm_entropy', 
                                      'spectral_entropy', 'svd_entropy']})
    return features

def compute_time_domain_features(signal: np.ndarray) -> Dict[str, float]:
    """Compute time domain features with validation"""
    features = {}
    try:
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['var'] = np.var(signal)
        features['skew'] = skew(signal) if len(signal) > 2 else 0
        features['kurtosis'] = kurtosis(signal) if len(signal) > 2 else 0
        features['zero_crossings'] = np.sum(np.diff(np.signbit(signal).astype(int)) != 0)
        features['peak_to_peak'] = np.max(signal) - np.min(signal)
        features['rms'] = np.sqrt(np.mean(signal ** 2))
    except Exception as e:
        raise FeatureExtractionError(f"Error in time domain feature computation: {str(e)}")
    return features

def compute_frequency_domain_features(signal: np.ndarray, fs: float = 250) -> Dict[str, float]:
    """Compute frequency domain features with validation"""
    features = {}
    try:
        freqs, psd = compute_psd(signal, fs)
        
        # Define standard EEG frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        # Compute band powers
        for band_name, band_range in bands.items():
            features[f'{band_name}_power'] = compute_band_power(freqs, psd, band_range)
        
        # Compute band power ratios
        alpha_power = features['alpha_power']
        beta_power = features['beta_power']
        theta_power = features['theta_power']
        
        if beta_power > 0:
            features['alpha_beta_ratio'] = alpha_power / beta_power
        else:
            features['alpha_beta_ratio'] = 0
            
        if theta_power > 0:
            features['beta_theta_ratio'] = beta_power / theta_power
        else:
            features['beta_theta_ratio'] = 0
        
        # Spectral entropy
        features['spectral_entropy'] = compute_spectral_entropy(psd)
        
        # Peak frequency
        peak_freq_idx = np.argmax(psd)
        features['peak_frequency'] = freqs[peak_freq_idx]
        
    except Exception as e:
        raise FeatureExtractionError(f"Error in frequency domain feature computation: {str(e)}")
    return features

def compute_wavelet_features(signal: np.ndarray, wavelet: str = 'db4', level: int = 4) -> Dict[str, float]:
    """Compute wavelet decomposition features"""
    features = {}
    try:
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        
        # Compute features for each decomposition level
        for i, coeff in enumerate(coeffs):
            # Energy
            features[f'wavelet_energy_level_{i}'] = np.sum(coeff ** 2)
            
            # Mean absolute value
            features[f'wavelet_mav_level_{i}'] = np.mean(np.abs(coeff))
            
            # Standard deviation
            features[f'wavelet_std_level_{i}'] = np.std(coeff)
            
            # Entropy
            hist, _ = np.histogram(coeff, bins=50, density=True)
            features[f'wavelet_entropy_level_{i}'] = -np.sum(hist * np.log2(hist + 1e-10))
        
        return features
    except Exception as e:
        logger.warning(f"Error in wavelet feature computation: {str(e)}")
        return {f'wavelet_{k}_level_{i}': 0 for i in range(level+1) for k in ['energy', 'mav', 'std', 'entropy']}

def compute_harmonic_features(signal: np.ndarray, fs: float = 250) -> Dict[str, float]:
    """Compute harmonic analysis features"""
    features = {}
    try:
        # Compute FFT
        n = len(signal)
        fft_vals = fft(signal)
        freqs = fftfreq(n, 1/fs)
        
        # Get positive frequencies only
        pos_freq_mask = freqs > 0
        freqs = freqs[pos_freq_mask]
        fft_vals = np.abs(fft_vals[pos_freq_mask])
        
        # Find peaks in frequency domain
        peaks, properties = find_peaks(fft_vals, height=0)
        
        if len(peaks) > 0:
            # Dominant frequency
            features['dominant_freq'] = freqs[peaks[np.argmax(properties['peak_heights'])]]
            
            # Harmonic frequencies
            peak_freqs = freqs[peaks]
            peak_heights = properties['peak_heights']
            
            # Sort peaks by height
            sorted_idx = np.argsort(peak_heights)[::-1]
            peak_freqs = peak_freqs[sorted_idx]
            peak_heights = peak_heights[sorted_idx]
            
            # Store top 3 harmonic frequencies and their amplitudes
            for i in range(min(3, len(peak_freqs))):
                features[f'harmonic_freq_{i+1}'] = peak_freqs[i]
                features[f'harmonic_amp_{i+1}'] = peak_heights[i]
        else:
            features.update({
                'dominant_freq': 0,
                'harmonic_freq_1': 0, 'harmonic_amp_1': 0,
                'harmonic_freq_2': 0, 'harmonic_amp_2': 0,
                'harmonic_freq_3': 0, 'harmonic_amp_3': 0
            })
        
        return features
    except Exception as e:
        logger.warning(f"Error in harmonic feature computation: {str(e)}")
        return {
            'dominant_freq': 0,
            'harmonic_freq_1': 0, 'harmonic_amp_1': 0,
            'harmonic_freq_2': 0, 'harmonic_amp_2': 0,
            'harmonic_freq_3': 0, 'harmonic_amp_3': 0
        }

def compute_phase_features(signal: np.ndarray) -> Dict[str, float]:
    """Compute phase-based features using Hilbert transform"""
    features = {}
    try:
        # Compute analytic signal
        analytic_signal = hilbert(signal)
        
        # Instantaneous phase
        phase = np.unwrap(np.angle(analytic_signal))
        
        # Phase features
        features['phase_mean'] = np.mean(phase)
        features['phase_std'] = np.std(phase)
        features['phase_range'] = np.max(phase) - np.min(phase)
        
        # Phase velocity
        phase_velocity = np.diff(phase)
        features['phase_velocity_mean'] = np.mean(phase_velocity)
        features['phase_velocity_std'] = np.std(phase_velocity)
        
        return features
    except Exception as e:
        logger.warning(f"Error in phase feature computation: {str(e)}")
        return {
            'phase_mean': 0, 'phase_std': 0, 'phase_range': 0,
            'phase_velocity_mean': 0, 'phase_velocity_std': 0
        }

def compute_cross_channel_features(signal1: np.ndarray, signal2: np.ndarray, fs: float = 250) -> Dict[str, float]:
    """Compute cross-channel features between two signals"""
    features = {}
    try:
        # Cross-correlation
        corr = np.correlate(signal1, signal2, mode='full')
        features['cross_corr_max'] = np.max(np.abs(corr))
        features['cross_corr_lag'] = np.argmax(np.abs(corr)) - len(signal1) + 1
        
        # Coherence
        freqs, coh = coherence(signal1, signal2, fs=fs)
        features['mean_coherence'] = np.mean(coh)
        features['max_coherence'] = np.max(coh)
        features['coherence_bandwidth'] = np.sum(coh > 0.5) * (freqs[1] - freqs[0])
        
        # Phase synchronization
        analytic1 = hilbert(signal1)
        analytic2 = hilbert(signal2)
        phase_diff = np.angle(analytic1) - np.angle(analytic2)
        features['phase_sync'] = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        return features
    except Exception as e:
        logger.warning(f"Error in cross-channel feature computation: {str(e)}")
        return {
            'cross_corr_max': 0, 'cross_corr_lag': 0,
            'mean_coherence': 0, 'max_coherence': 0, 'coherence_bandwidth': 0,
            'phase_sync': 0
        }

def compute_pca_features(signals: np.ndarray, n_components: int = 3) -> Dict[str, float]:
    """Compute PCA-based features from multiple channels"""
    features = {}
    try:
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca.fit(signals)
        
        # Store explained variance ratios
        for i, var_ratio in enumerate(pca.explained_variance_ratio_):
            features[f'pca_var_ratio_{i+1}'] = var_ratio
        
        # Store total explained variance
        features['pca_total_var_ratio'] = np.sum(pca.explained_variance_ratio_)
        
        return features
    except Exception as e:
        logger.warning(f"Error in PCA feature computation: {str(e)}")
        return {f'pca_var_ratio_{i+1}': 0 for i in range(n_components)}

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced feature extraction with advanced EEG-specific features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing EEG data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing extracted features
    """
    try:
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        feature_list = []
        
        def process_channel(channel_data: np.ndarray, col_name: str) -> Dict[str, float]:
            """Process a single channel's data"""
            features = {}
            
            # Time domain features
            features.update({f'{col_name}_{k}': v for k, v in 
                           compute_time_domain_features(channel_data).items()})
            
            # Hjorth parameters
            activity, mobility, complexity = compute_hjorth_parameters(channel_data)
            features[f'{col_name}_activity'] = activity
            features[f'{col_name}_mobility'] = mobility
            features[f'{col_name}_complexity'] = complexity
            
            # Frequency domain features
            freq_features = compute_frequency_domain_features(channel_data)
            features.update({f'{col_name}_{k}': v for k, v in freq_features.items()})
            
            # Nonlinear features
            nl_features = compute_nonlinear_features(channel_data)
            features.update({f'{col_name}_{k}': v for k, v in nl_features.items()})
            
            # Wavelet features
            wavelet_features = compute_wavelet_features(channel_data)
            features.update({f'{col_name}_{k}': v for k, v in wavelet_features.items()})
            
            # Harmonic features
            harmonic_features = compute_harmonic_features(channel_data)
            features.update({f'{col_name}_{k}': v for k, v in harmonic_features.items()})
            
            # Phase features
            phase_features = compute_phase_features(channel_data)
            features.update({f'{col_name}_{k}': v for k, v in phase_features.items()})
            
            return features
        
        # Process each row
        for i, row in df.iterrows():
            feature_data = {}
            
            # Process each channel
            for col in numerical_cols:
                if col != 'eeg_state':
                    signal = np.array([row[col]]) if isinstance(row[col], (int, float)) else row[col]
                    feature_data.update(process_channel(signal, col))
            
            # Multi-channel features
            eeg_channels = [col for col in numerical_cols if col != 'eeg_state']
            if len(eeg_channels) > 1:
                # Compute cross-channel features
                for i, ch1 in enumerate(eeg_channels[:-1]):
                    for ch2 in eeg_channels[i+1:]:
                        sig1 = np.array([row[ch1]]) if isinstance(row[ch1], (int, float)) else row[ch1]
                        sig2 = np.array([row[ch2]]) if isinstance(row[ch2], (int, float)) else row[ch2]
                        
                        if len(sig1) > 1 and len(sig2) > 1:
                            try:
                                cross_features = compute_cross_channel_features(sig1, sig2)
                                feature_data.update({
                                    f'{ch1}_{ch2}_{k}': v for k, v in cross_features.items()
                                })
                            except Exception:
                                pass
                
                # Compute PCA features for all channels
                try:
                    signals = np.array([row[col] for col in eeg_channels])
                    pca_features = compute_pca_features(signals)
                    feature_data.update(pca_features)
                except Exception:
                    pass
            
            # Keep original label
            if 'eeg_state' in row:
                feature_data['eeg_state'] = row['eeg_state']
            
            feature_list.append(feature_data)
        
        return pd.DataFrame(feature_list)
        
    except Exception as e:
        raise FeatureExtractionError(f"Error in feature extraction: {str(e)}")
