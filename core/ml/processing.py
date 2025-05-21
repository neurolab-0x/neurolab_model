import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
import logging
import json
from ..models.eeg import EEGFeatures

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