import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
import logging
import json
from ..models.eeg import EEGFeatures, EEGChannelData, EEGRecording
from scipy import signal

logger = logging.getLogger(__name__)

def load_data(file_path: str) -> np.ndarray:
    """Load EEG data from file"""
    try:
        logger.info(f"Loading data from file: {file_path}")
        if file_path.endswith('.csv'):
            # Read CSV with pandas to handle metadata column
            logger.debug("Reading CSV file with pandas")
            df = pd.read_csv(file_path)
            logger.debug(f"CSV columns: {df.columns.tolist()}")
            
            # Extract only the numerical columns (alpha, beta, theta, delta, gamma)
            logger.debug("Extracting numerical columns")
            data = df[['alpha', 'beta', 'theta', 'delta', 'gamma']].values
            logger.debug(f"Extracted data shape: {data.shape}")
            return data
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise

def label_eeg_states(data: np.ndarray) -> np.ndarray:
    """Label EEG states based on frequency band ratios"""
    try:
        logger.info("Labeling EEG states")
        # Extract frequency bands (assuming columns are in order: alpha, beta, theta, delta, gamma)
        alpha = data[:, 0]
        beta = data[:, 1]
        theta = data[:, 2]
        delta = data[:, 3]
        gamma = data[:, 4]
        
        logger.debug(f"Frequency band shapes - alpha: {alpha.shape}, beta: {beta.shape}, theta: {theta.shape}")
        
        # Calculate ratios
        beta_alpha_ratio = beta / (alpha + 1e-10)
        theta_beta_ratio = theta / (beta + 1e-10)
        
        # Label states based on ratios
        states = np.zeros(len(data))
        
        # Relaxation: high alpha, low beta
        states[beta_alpha_ratio < 0.5] = 0
        
        # Attention: high beta, low theta
        states[(beta_alpha_ratio > 1.2) & (theta_beta_ratio < 0.5)] = 1
        
        # Stress: high beta, high theta
        states[(beta_alpha_ratio > 1.2) & (theta_beta_ratio > 0.8)] = 2
        
        logger.debug(f"Labeled states distribution: {np.bincount(states.astype(int))}")
        return states
    except Exception as e:
        logger.error(f"Error labeling EEG states: {str(e)}")
        raise

def extract_features(data: np.ndarray) -> List[EEGFeatures]:
    """Extract EEG features from raw data"""
    try:
        logger.info("Extracting EEG features")
        features = []
        for i, row in enumerate(data):
            try:
                logger.debug(f"Processing row {i}: {row}")
                feature = EEGFeatures(
                    alpha=float(row[0]),
                    beta=float(row[1]),
                    theta=float(row[2]),
                    delta=float(row[3]),
                    gamma=float(row[4])
                )
                features.append(feature)
            except Exception as row_error:
                logger.error(f"Error processing row {i}: {str(row_error)}")
                raise
        logger.debug(f"Extracted {len(features)} features")
        return features
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise

def preprocess_data(features: List[EEGFeatures]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess EEG features for model input"""
    try:
        logger.info("Preprocessing features")
        # Convert features to numpy array
        logger.debug("Converting features to numpy array")
        X = np.array([[f.alpha, f.beta, f.theta, f.delta, f.gamma] for f in features])
        logger.debug(f"Feature array shape: {X.shape}")
        
        # Normalize features
        logger.debug("Normalizing features")
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / (X_std + 1e-10)
        logger.debug(f"Normalized feature shape: {X_normalized.shape}")
        
        # Create labels (placeholder - should be replaced with actual labels)
        y = np.zeros(len(features))
        
        # Split into train/test sets (80/20)
        split_idx = int(0.8 * len(X_normalized))
        X_train = X_normalized[:split_idx]
        X_test = X_normalized[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        logger.debug(f"Train set size: {X_train.shape}, Test set size: {X_test.shape}")
        return X_normalized, X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def temporal_smoothing(predictions: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply temporal smoothing to state predictions"""
    try:
        smoothed = np.zeros_like(predictions)
        for i in range(len(predictions)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(predictions), i + window_size // 2 + 1)
            window = predictions[start_idx:end_idx]
            smoothed[i] = np.argmax(np.bincount(window.astype(int)))
        return smoothed
    except Exception as e:
        logger.error(f"Error in temporal smoothing: {str(e)}")
        raise

def calculate_state_durations(states: np.ndarray) -> Dict[int, float]:
    """Calculate duration of each state"""
    try:
        durations = {}
        current_state = states[0]
        duration = 1
        
        for state in states[1:]:
            if state == current_state:
                duration += 1
            else:
                durations[current_state] = durations.get(current_state, 0) + duration
                current_state = state
                duration = 1
        
        # Add the last state
        durations[current_state] = durations.get(current_state, 0) + duration
        
        return durations
    except Exception as e:
        logger.error(f"Error calculating state durations: {str(e)}")
        raise

def generate_recommendations(state_durations: Dict[int, float], total_duration: float, confidence: float) -> List[Dict[str, Any]]:
    """Generate clinical recommendations based on state analysis"""
    try:
        recommendations = []
        
        # Calculate state percentages
        state_percentages = {
            state: duration / total_duration
            for state, duration in state_durations.items()
        }
        
        # Stress analysis
        if 2 in state_percentages and state_percentages[2] > 0.3:
            recommendations.append({
                "type": "stress_management",
                "severity": "high" if state_percentages[2] > 0.5 else "medium",
                "message": "Elevated stress levels detected. Consider stress reduction techniques.",
                "confidence": confidence
            })
        
        # Attention analysis
        if 1 in state_percentages and state_percentages[1] < 0.2:
            recommendations.append({
                "type": "attention_improvement",
                "severity": "medium",
                "message": "Low attention levels observed. Consider attention training exercises.",
                "confidence": confidence
            })
        
        # Relaxation analysis
        if 0 in state_percentages and state_percentages[0] < 0.2:
            recommendations.append({
                "type": "relaxation_techniques",
                "severity": "medium",
                "message": "Limited relaxation periods detected. Consider relaxation exercises.",
                "confidence": confidence
            })
        
        return recommendations
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise

def load_raw_eeg_data(file_path: str) -> EEGRecording:
    """Load raw EEG data from various file formats"""
    try:
        logger.info(f"Loading raw EEG data from file: {file_path}")
        
        if file_path.endswith('.edf'):
            return _load_edf_data(file_path)
        elif file_path.endswith('.csv'):
            return _load_csv_data(file_path)
        elif file_path.endswith('.bdf'):
            return _load_bdf_data(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
            
    except Exception as e:
        logger.error(f"Error loading raw EEG data: {str(e)}")
        raise

def _load_csv_data(file_path: str) -> EEGRecording:
    """Load raw EEG data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        
        # Extract channel data
        channels = []
        for col in df.columns:
            if col not in ['timestamp', 'subject_id', 'session_id']:
                channel_data = EEGChannelData(
                    channel_name=col,
                    sampling_rate=_estimate_sampling_rate(df['timestamp']),
                    data=df[col].values,
                    metadata={'source': 'csv'}
                )
                channels.append(channel_data)
        
        # Create recording
        recording = EEGRecording(
            channels=channels,
            start_time=pd.to_datetime(df['timestamp'].iloc[0]),
            end_time=pd.to_datetime(df['timestamp'].iloc[-1]),
            subject_id=df['subject_id'].iloc[0],
            session_id=df['session_id'].iloc[0],
            metadata={'source_file': file_path}
        )
        
        return recording
        
    except Exception as e:
        logger.error(f"Error loading CSV data: {str(e)}")
        raise

def _estimate_sampling_rate(timestamps: pd.Series) -> float:
    """Estimate sampling rate from timestamps"""
    try:
        # Convert to datetime if needed
        if isinstance(timestamps.iloc[0], str):
            timestamps = pd.to_datetime(timestamps)
        
        # Calculate time differences
        time_diffs = np.diff(timestamps.astype(np.int64) // 10**9)  # Convert to seconds
        return 1.0 / np.median(time_diffs)
    except Exception as e:
        logger.warning(f"Error estimating sampling rate: {str(e)}")
        return 250.0  # Default to 250 Hz

def preprocess_raw_eeg(recording: EEGRecording,
                      notch_freq: float = 50.0,
                      bandpass_low: float = 0.5,
                      bandpass_high: float = 45.0) -> EEGRecording:
    """Preprocess raw EEG data with standard filters"""
    try:
        logger.info("Preprocessing raw EEG data")
        processed_channels = []
        
        for channel in recording.channels:
            # Apply notch filter for power line noise
            data = _apply_notch_filter(channel.data, channel.sampling_rate, notch_freq)
            
            # Apply bandpass filter
            data = _apply_bandpass_filter(data, channel.sampling_rate, bandpass_low, bandpass_high)
            
            # Create new channel with processed data
            processed_channel = EEGChannelData(
                channel_name=channel.channel_name,
                sampling_rate=channel.sampling_rate,
                data=data,
                metadata={
                    **channel.metadata,
                    'preprocessing': {
                        'notch_freq': notch_freq,
                        'bandpass_low': bandpass_low,
                        'bandpass_high': bandpass_high
                    }
                }
            )
            processed_channels.append(processed_channel)
        
        # Create new recording with processed channels
        processed_recording = EEGRecording(
            channels=processed_channels,
            start_time=recording.start_time,
            end_time=recording.end_time,
            subject_id=recording.subject_id,
            session_id=recording.session_id,
            metadata={
                **recording.metadata,
                'preprocessing_applied': True
            }
        )
        
        return processed_recording
        
    except Exception as e:
        logger.error(f"Error preprocessing raw EEG data: {str(e)}")
        raise

def _apply_notch_filter(data: np.ndarray, fs: float, notch_freq: float) -> np.ndarray:
    """Apply notch filter to remove power line noise"""
    try:
        # Design notch filter
        nyquist = fs / 2
        freq = notch_freq / nyquist
        b, a = signal.iirnotch(freq, 30.0)
        
        # Apply filter
        return signal.filtfilt(b, a, data)
    except Exception as e:
        logger.error(f"Error applying notch filter: {str(e)}")
        return data

def _apply_bandpass_filter(data: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    """Apply bandpass filter to isolate EEG frequency range"""
    try:
        # Design bandpass filter
        nyquist = fs / 2
        low = low / nyquist
        high = high / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter
        return signal.filtfilt(b, a, data)
    except Exception as e:
        logger.error(f"Error applying bandpass filter: {str(e)}")
        return data

def extract_frequency_bands(recording: EEGRecording,
                          window_size: int = 256,
                          overlap: int = 128) -> List[EEGFeatures]:
    """Extract frequency band features from raw EEG data"""
    try:
        logger.info("Extracting frequency band features")
        features = []
        
        for channel in recording.channels:
            # Calculate spectrogram
            freqs, times, Sxx = signal.spectrogram(
                channel.data,
                fs=channel.sampling_rate,
                window='hann',
                nperseg=window_size,
                noverlap=overlap
            )
            
            # Define frequency bands
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 45)
            }
            
            # Calculate band powers
            for time_idx, time in enumerate(times):
                band_powers = {}
                for band_name, (low, high) in bands.items():
                    # Find frequency indices for this band
                    band_mask = (freqs >= low) & (freqs <= high)
                    # Calculate mean power in band
                    band_powers[band_name] = np.mean(Sxx[band_mask, time_idx])
                
                # Create features
                feature = EEGFeatures(
                    alpha=band_powers['alpha'],
                    beta=band_powers['beta'],
                    theta=band_powers['theta'],
                    delta=band_powers['delta'],
                    gamma=band_powers['gamma'],
                    signal_quality=_calculate_signal_quality(band_powers),
                    metadata={
                        'channel': channel.channel_name,
                        'timestamp': recording.start_time + pd.Timedelta(seconds=time)
                    }
                )
                features.append(feature)
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting frequency bands: {str(e)}")
        raise

def _calculate_signal_quality(band_powers: Dict[str, float]) -> float:
    """Calculate signal quality based on frequency band powers"""
    try:
        total_power = sum(band_powers.values())
        if total_power == 0:
            return 0.0
            
        # Calculate power distribution
        power_dist = {band: power/total_power for band, power in band_powers.items()}
        
        # Ideal distribution (can be adjusted based on expected patterns)
        ideal_dist = {
            'delta': 0.2,
            'theta': 0.2,
            'alpha': 0.3,
            'beta': 0.2,
            'gamma': 0.1
        }
        
        # Calculate quality as inverse of deviation from ideal
        deviation = sum(abs(power_dist[band] - ideal_dist[band]) for band in band_powers)
        quality = 1.0 - min(deviation, 1.0)
        
        return quality
        
    except Exception as e:
        logger.error(f"Error calculating signal quality: {str(e)}")
        return 0.0 