import numpy as np
from scipy import signal, stats
import logging
from typing import Dict, List, Tuple, Union
import pandas as pd

from utils.artifacts import detect_outliers, detect_muscle_artifacts, detect_eye_blinks

logger = logging.getLogger(__name__)

def calculate_snr(eeg_signal: np.ndarray, fs: float = 250.0) -> float:
    """
    Calculate signal-to-noise ratio for EEG signal
    
    Parameters:
    -----------
    eeg_signal : np.ndarray
        EEG signal data
    fs : float
        Sampling frequency in Hz
        
    Returns:
    --------
    float : Signal-to-noise ratio in dB
    """
    # Use Welch's method to estimate PSD
    freqs, psd = signal.welch(eeg_signal, fs=fs, nperseg=min(256, len(eeg_signal)))
    
    # Define signal and noise frequency bands
    # Signal: Alpha (8-13 Hz) + Beta (13-30 Hz) bands
    signal_mask = (freqs >= 8) & (freqs <= 30)
    # Noise: High frequency components (>45 Hz) and very low frequency drift (<0.5 Hz)
    noise_mask = (freqs < 0.5) | (freqs > 45)
    
    # Calculate power in each band
    signal_power = np.sum(psd[signal_mask])
    noise_power = np.sum(psd[noise_mask])
    
    # Calculate SNR in dB (avoid division by zero)
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = 100.0  # Set to high value if no noise is detected
        
    return snr

def assess_electrode_contact(eeg_signal: np.ndarray, threshold: float = 0.95) -> bool:
    """
    Assess electrode contact quality based on signal variance and flatline detection
    
    Parameters:
    -----------
    eeg_signal : np.ndarray
        EEG signal data
    threshold : float
        Similarity threshold for flatline detection
        
    Returns:
    --------
    bool : True if good contact, False if poor contact
    """
    # Check for flatlines (consecutive identical or near-identical values)
    if len(eeg_signal) > 10:
        # Calculate differences between consecutive samples
        diffs = np.abs(np.diff(eeg_signal))
        
        # Check percentage of differences that are near zero
        zero_count = np.sum(diffs < (np.std(eeg_signal) * 0.01))
        zero_percent = zero_count / (len(eeg_signal) - 1)
        
        # Poor contact if many consecutive values are nearly identical
        if zero_percent > threshold:
            return False
    
    # Check for very low signal variance (dead channel)
    if np.var(eeg_signal) < 1e-6:
        return False
        
    return True

def detect_line_noise(eeg_signal: np.ndarray, fs: float = 250.0, 
                     line_freq: float = 50.0, threshold: float = 0.3) -> bool:
    """
    Detect power line interference (50 Hz or 60 Hz)
    
    Parameters:
    -----------
    eeg_signal : np.ndarray
        EEG signal data
    fs : float
        Sampling frequency in Hz
    line_freq : float
        Line frequency to detect (typically 50 or 60 Hz)
    threshold : float
        Power ratio threshold for detection
        
    Returns:
    --------
    bool : True if significant line noise detected
    """
    # Calculate PSD
    freqs, psd = signal.welch(eeg_signal, fs=fs, nperseg=min(512, len(eeg_signal)))
    
    # Find the index closest to line frequency
    line_idx = np.argmin(np.abs(freqs - line_freq))
    
    # Calculate power in line frequency band (±1Hz)
    band_min = max(0, line_idx - int(1 * len(freqs) / fs))
    band_max = min(len(freqs) - 1, line_idx + int(1 * len(freqs) / fs))
    line_power = np.sum(psd[band_min:band_max])
    
    # Calculate total power
    total_power = np.sum(psd)
    
    # Detect if line noise is significant
    return (line_power / total_power) > threshold if total_power > 0 else False

def calculate_stationarity(eeg_signal: np.ndarray, window_size: int = 100) -> float:
    """
    Calculate signal stationarity using variance of windowed means
    
    Parameters:
    -----------
    eeg_signal : np.ndarray
        EEG signal data
    window_size : int
        Size of windows for stationarity analysis
        
    Returns:
    --------
    float : Stationarity measure (lower values indicate more stationary signal)
    """
    if len(eeg_signal) < window_size * 2:
        return 0.0
    
    # Create windows
    n_windows = len(eeg_signal) // window_size
    windows = np.array_split(eeg_signal[:n_windows * window_size], n_windows)
    
    # Calculate mean of each window
    window_means = np.array([np.mean(w) for w in windows])
    
    # Calculate variance of window means, normalized by signal variance
    signal_var = np.var(eeg_signal)
    if signal_var > 0:
        stationarity = np.var(window_means) / signal_var
    else:
        stationarity = 0.0
        
    return stationarity

def assess_channel_quality(eeg_signal: np.ndarray, fs: float = 250.0) -> Dict[str, float]:
    """
    Comprehensive quality assessment for a single EEG channel
    
    Parameters:
    -----------
    eeg_signal : np.ndarray
        EEG signal data for one channel
    fs : float
        Sampling frequency in Hz
        
    Returns:
    --------
    dict : Dictionary of quality metrics
    """
    metrics = {}
    
    # Basic signal properties
    metrics['mean'] = float(np.mean(eeg_signal))
    metrics['variance'] = float(np.var(eeg_signal))
    metrics['range'] = float(np.ptp(eeg_signal))
    
    # SNR
    metrics['snr'] = float(calculate_snr(eeg_signal, fs))
    
    # Contact quality (binary)
    metrics['good_contact'] = bool(assess_electrode_contact(eeg_signal))
    
    # Line noise detection
    metrics['line_noise_50hz'] = bool(detect_line_noise(eeg_signal, fs, 50.0))
    metrics['line_noise_60hz'] = bool(detect_line_noise(eeg_signal, fs, 60.0))
    
    # Artifact percentage
    outlier_mask = detect_outliers(eeg_signal)
    muscle_mask = detect_muscle_artifacts(eeg_signal, fs)
    eye_mask = detect_eye_blinks(eeg_signal, fs)
    
    combined_artifact_mask = outlier_mask | muscle_mask | eye_mask
    metrics['artifact_percentage'] = float(np.mean(combined_artifact_mask) * 100.0)
    
    # Stationarity
    metrics['non_stationarity'] = float(calculate_stationarity(eeg_signal))
    
    # Overall quality score (0-100)
    quality_score = 100.0
    
    # Penalize for artifacts
    quality_score -= min(50, metrics['artifact_percentage'])
    
    # Penalize for poor SNR (SNR below 5dB is considered poor)
    if metrics['snr'] < 5:
        quality_score -= min(30, (5 - metrics['snr']) * 6)
    
    # Penalize for poor contact
    if not metrics['good_contact']:
        quality_score -= 40
    
    # Penalize for line noise
    if metrics['line_noise_50hz'] or metrics['line_noise_60hz']:
        quality_score -= 10
    
    # Penalize for non-stationarity
    if metrics['non_stationarity'] > 0.5:
        quality_score -= min(20, metrics['non_stationarity'] * 20)
    
    # Clip to 0-100 range
    metrics['quality_score'] = max(0, min(100, quality_score))
    
    return metrics

def evaluate_eeg_quality(eeg_data: np.ndarray, fs: float = 250.0, 
                        channel_names: List[str] = None) -> Dict:
    """
    Evaluate quality of multi-channel EEG data
    
    Parameters:
    -----------
    eeg_data : np.ndarray
        EEG data with shape (channels, samples)
    fs : float
        Sampling frequency in Hz
    channel_names : list or None
        Optional list of channel names
        
    Returns:
    --------
    dict : Dictionary containing quality report for each channel and overall dataset
    """
    if eeg_data.ndim == 1:
        eeg_data = eeg_data.reshape(1, -1)
    
    n_channels = eeg_data.shape[0]
    
    if channel_names is None:
        channel_names = [f"channel_{i}" for i in range(n_channels)]
    
    # Evaluate each channel
    channel_reports = {}
    for i, channel_name in enumerate(channel_names):
        try:
            if i < n_channels:
                channel_reports[channel_name] = assess_channel_quality(eeg_data[i], fs)
            else:
                logger.warning(f"Channel name {channel_name} provided but only {n_channels} channels in data")
        except Exception as e:
            logger.error(f"Error assessing quality for channel {channel_name}: {str(e)}")
            channel_reports[channel_name] = {'error': str(e)}
    
    # Overall dataset quality
    overall_quality = {}
    
    # Average metrics across channels
    quality_scores = [report.get('quality_score', 0) for report in channel_reports.values() 
                     if isinstance(report, dict) and 'quality_score' in report]
    
    if quality_scores:
        overall_quality['mean_quality_score'] = float(np.mean(quality_scores))
        overall_quality['min_quality_score'] = float(np.min(quality_scores))
        overall_quality['channel_count'] = n_channels
        overall_quality['poor_quality_channels'] = sum(1 for score in quality_scores if score < 50)
    
    # Overall dataset assessment
    if overall_quality.get('mean_quality_score', 0) < 40:
        overall_quality['quality_assessment'] = "Poor - significant data quality issues"
    elif overall_quality.get('mean_quality_score', 0) < 70:
        overall_quality['quality_assessment'] = "Moderate - some channels may need attention"
    else:
        overall_quality['quality_assessment'] = "Good - acceptable quality for analysis"
        
    # Specific recommendations
    recommendations = []
    
    if overall_quality.get('poor_quality_channels', 0) > n_channels / 3:
        recommendations.append("Many channels have poor quality. Consider redoing recording or extensive preprocessing.")
    
    if any(report.get('line_noise_50hz', False) or report.get('line_noise_60hz', False) 
          for report in channel_reports.values() if isinstance(report, dict)):
        recommendations.append("Line noise detected. Consider applying a notch filter at 50/60Hz.")
    
    if any(not report.get('good_contact', True) for report in channel_reports.values() if isinstance(report, dict)):
        recommendations.append("Poor electrode contact detected. Check electrode placement and impedance.")
        
    overall_quality['recommendations'] = recommendations
    
    return {
        'channel_reports': channel_reports,
        'overall_quality': overall_quality,
    }

def select_best_channels(eeg_data: np.ndarray, fs: float = 250.0, 
                       channel_names: List[str] = None, select_top_n: int = None,
                       quality_threshold: float = 60.0) -> List[str]:
    """
    Select most informative EEG channels based on quality metrics and signal characteristics
    
    Parameters:
    -----------
    eeg_data : np.ndarray
        EEG data with shape (channels, samples)
    fs : float
        Sampling frequency in Hz
    channel_names : list or None
        Optional list of channel names
    select_top_n : int or None
        Number of top channels to select (if None, uses quality_threshold)
    quality_threshold : float
        Minimum quality score to include channel
    
    Returns:
    --------
    list : Selected channel names
    """
    if eeg_data.ndim == 1:
        eeg_data = eeg_data.reshape(1, -1)
    
    n_channels = eeg_data.shape[0]
    
    if channel_names is None:
        channel_names = [f"channel_{i}" for i in range(n_channels)]
        
    # Get quality assessment
    quality_report = evaluate_eeg_quality(eeg_data, fs, channel_names)
    channel_reports = quality_report['channel_reports']
    
    # Create a list of (channel_name, quality_score) tuples
    channel_quality = []
    for ch_name, report in channel_reports.items():
        if isinstance(report, dict) and 'quality_score' in report:
            channel_quality.append((ch_name, report['quality_score']))
    
    # Sort by quality score (highest first)
    channel_quality.sort(key=lambda x: x[1], reverse=True)
    
    # Select channels
    if select_top_n is not None:
        # Select top N channels
        selected = [ch for ch, _ in channel_quality[:select_top_n]]
    else:
        # Select channels above quality threshold
        selected = [ch for ch, score in channel_quality if score >= quality_threshold]
    
    return selected

def calculate_channel_importance(eeg_data: np.ndarray, labels: np.ndarray = None,
                               channel_names: List[str] = None) -> Dict[str, float]:
    """
    Calculate channel importance using statistical measures or ML feature importance
    
    Parameters:
    -----------
    eeg_data : np.ndarray
        EEG data with shape (channels, samples) or (samples, channels)
    labels : np.ndarray or None
        Optional class labels for supervised importance
    channel_names : list or None
        Optional list of channel names
        
    Returns:
    --------
    dict : Channel importance scores
    """
    # Ensure data is in (channels, samples) format
    if eeg_data.shape[0] > eeg_data.shape[1]:  # More samples than channels
        eeg_data = eeg_data.T
    
    n_channels = eeg_data.shape[0]
    
    if channel_names is None:
        channel_names = [f"channel_{i}" for i in range(n_channels)]
    
    importance_scores = {}
    
    if labels is not None and len(labels) == eeg_data.shape[1]:
        # Supervised importance using ANOVA F-value
        from sklearn.feature_selection import f_classif
        
        # Calculate feature statistics for each channel
        for i, ch_name in enumerate(channel_names):
            if i < n_channels:
                try:
                    # Reshape to 2D for sklearn
                    X = eeg_data[i].reshape(-1, 1)
                    f_values, p_values = f_classif(X, labels)
                    importance_scores[ch_name] = float(f_values[0]) if not np.isnan(f_values[0]) else 0.0
                except Exception as e:
                    logger.warning(f"Error calculating importance for {ch_name}: {str(e)}")
                    importance_scores[ch_name] = 0.0
    else:
        # Unsupervised importance based on signal characteristics
        for i, ch_name in enumerate(channel_names):
            if i < n_channels:
                try:
                    # Signal variance - higher variance often means more information
                    var_score = np.var(eeg_data[i])
                    
                    # SNR - higher SNR means better signal quality
                    snr_score = calculate_snr(eeg_data[i])
                    
                    # Spectral entropy - higher entropy often means more information
                    from scipy.stats import entropy
                    freqs, psd = signal.welch(eeg_data[i], nperseg=min(256, eeg_data.shape[1]))
                    spectral_entropy = entropy(psd + 1e-10)  # Add small constant to avoid log(0)
                    
                    # Combine metrics (with some weights)
                    importance = 0.4 * var_score + 0.3 * snr_score + 0.3 * spectral_entropy
                    importance_scores[ch_name] = float(importance)
                except Exception as e:
                    logger.warning(f"Error calculating importance for {ch_name}: {str(e)}")
                    importance_scores[ch_name] = 0.0
    
    # Normalize to 0-100 scale
    if importance_scores:
        max_score = max(importance_scores.values())
        if max_score > 0:
            importance_scores = {k: (v / max_score) * 100 for k, v in importance_scores.items()}
    
    return importance_scores

def generate_channel_selection_report(eeg_data: np.ndarray, fs: float = 250.0, 
                                   labels: np.ndarray = None,
                                   channel_names: List[str] = None,
                                   select_top_n: int = None) -> Dict:
    """
    Generate a comprehensive channel selection report combining quality and importance
    
    Parameters:
    -----------
    eeg_data : np.ndarray
        EEG data with shape (channels, samples)
    fs : float
        Sampling frequency in Hz
    labels : np.ndarray or None
        Optional class labels for supervised importance
    channel_names : list or None
        Optional list of channel names
    select_top_n : int or None
        Number of top channels to select
        
    Returns:
    --------
    dict : Channel selection report with recommendations
    """
    if eeg_data.ndim == 1:
        eeg_data = eeg_data.reshape(1, -1)
    
    n_channels = eeg_data.shape[0]
    
    if channel_names is None:
        channel_names = [f"channel_{i}" for i in range(n_channels)]
    
    # Get quality assessment
    quality_report = evaluate_eeg_quality(eeg_data, fs, channel_names)
    channel_reports = quality_report['channel_reports']
    
    # Get channel importance
    importance_scores = calculate_channel_importance(eeg_data, labels, channel_names)
    
    # Combined score (quality + importance)
    combined_scores = {}
    for ch_name in channel_names:
        if ch_name in channel_reports and isinstance(channel_reports[ch_name], dict):
            quality_score = channel_reports[ch_name].get('quality_score', 0)
            importance = importance_scores.get(ch_name, 0)
            # Weighted combination (quality is more important)
            combined_scores[ch_name] = 0.7 * quality_score + 0.3 * importance
    
    # Sort channels by combined score
    ranked_channels = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Select top N channels if specified
    if select_top_n is not None:
        selected_channels = [ch for ch, _ in ranked_channels[:select_top_n]]
    else:
        # Otherwise select channels with score above 50
        selected_channels = [ch for ch, score in ranked_channels if score >= 50]
    
    # Generate report
    report = {
        'selected_channels': selected_channels,
        'channel_rankings': [
            {
                'channel_name': ch,
                'combined_score': combined,
                'quality_score': channel_reports.get(ch, {}).get('quality_score', 0),
                'importance_score': importance_scores.get(ch, 0)
            }
            for ch, combined in ranked_channels
        ],
        'selection_criteria': {
            'method': 'Combined quality and information content',
            'quality_weight': 0.7,
            'importance_weight': 0.3
        }
    }
    
    return report 