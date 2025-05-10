import numpy as np
import logging
import time
from collections import deque
from threading import Lock
from preprocessing.preprocess import preprocess_data
from preprocessing.features import extract_features
from utils.model_loading import load_calibrated_model
from utils.temporal_processing import temporal_smoothing
from utils.artifacts import clean_eeg
from utils.filters import apply_eeg_preprocessing
from config.settings import PROCESSING_CONFIG

logger = logging.getLogger(__name__)

# Singleton model cache to prevent redundant model loading
class ModelCache:
    _instance = None
    _lock = Lock()
    _loaded_models = {}
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance
    
    def get_model(self, model_path):
        """Get model from cache or load it if not available"""
        if model_path not in self._loaded_models:
            logger.info(f"Loading model from {model_path} (not in cache)")
            self._loaded_models[model_path] = load_calibrated_model(model_path)
        return self._loaded_models[model_path]
    
    def clear_cache(self):
        """Clear model cache"""
        self._loaded_models = {}


# Streaming buffer for continuous data processing
class StreamBuffer:
    def __init__(self, max_size=5000, channels=None):
        """Initialize a streaming buffer for EEG data
        
        Parameters:
        -----------
        max_size : int
            Maximum number of samples to store in buffer
        channels : int or None
            Number of EEG channels
        """
        self.max_size = max_size
        self.channels = channels
        self.buffer = None
        self.last_processed_idx = 0
        
    def add_data(self, data):
        """Add new data to the buffer
        
        Parameters:
        -----------
        data : array-like
            New EEG data samples
            
        Returns:
        --------
        new_data_range : tuple
            (start_idx, end_idx) of newly added data
        """
        # Convert to numpy array if needed
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Handle dimensionality
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        # Initialize buffer if needed
        if self.buffer is None:
            self.channels = data.shape[0]
            self.buffer = data
            return (0, data.shape[1])
        
        # Append new data
        self.buffer = np.hstack([self.buffer, data])
        
        # Trim buffer if it exceeds max size
        if self.buffer.shape[1] > self.max_size:
            trim_size = self.buffer.shape[1] - self.max_size
            self.buffer = self.buffer[:, trim_size:]
            self.last_processed_idx = max(0, self.last_processed_idx - trim_size)
            
        # Return range of new data
        return (self.buffer.shape[1] - data.shape[1], self.buffer.shape[1])
    
    def get_unprocessed_data(self):
        """Get data that hasn't been processed yet
        
        Returns:
        --------
        data : array-like
            Unprocessed data
        """
        if self.buffer is None or self.last_processed_idx >= self.buffer.shape[1]:
            return None
            
        data = self.buffer[:, self.last_processed_idx:]
        self.last_processed_idx = self.buffer.shape[1]
        return data
    
    def get_window(self, window_size=None):
        """Get the most recent window of data
        
        Parameters:
        -----------
        window_size : int or None
            Size of window to return, if None uses all available data
            
        Returns:
        --------
        data : array-like
            Windowed data
        """
        if self.buffer is None:
            return None
            
        if window_size is None or window_size >= self.buffer.shape[1]:
            return self.buffer
            
        return self.buffer[:, -window_size:]


# Calculate adaptive window size based on data characteristics
def calculate_adaptive_window(data, min_window=50, max_window=500):
    """Calculate adaptive window size based on signal characteristics
    
    Parameters:
    -----------
    data : array-like
        EEG data
    min_window : int
        Minimum window size
    max_window : int
        Maximum window size
        
    Returns:
    --------
    window_size : int
        Calculated window size
    """
    if data is None or data.size == 0:
        return min_window
        
    # Use the amplitude variance to determine stability
    variance = np.var(data)
    
    # Higher variance (less stable) = smaller window
    # Lower variance (more stable) = larger window
    if variance > 100:  # High variance
        return min_window
    elif variance < 10:  # Low variance
        return max_window
    else:
        # Scale linearly between min and max window
        norm_var = (variance - 10) / 90  # Normalize between 0 and 1
        window_size = max_window - norm_var * (max_window - min_window)
        return int(window_size)


def process_realtime_data(data, model=None, model_path=None, clean_artifacts=True, stream_buffer=None):
    """
    Processes incoming real-time EEG data and returns model predictions with detailed metrics.
    
    Parameters:
    -----------
    data : dict or array-like
        Raw EEG data in numpy array, list format, or dict with channels
    model : keras.Model or None
        Pre-loaded model to use for predictions, if None will load from model_path
    model_path : str or None
        Path to model file if model is not provided
    clean_artifacts : bool
        Whether to apply artifact removal to the signal
    stream_buffer : StreamBuffer or None
        Optional buffer for continuous data processing
        
    Returns:
    --------
    dict : Prediction results with state, confidence, and additional metrics
    """
    start_time = time.time()
    
    try:
        # Convert input to numpy array
        if isinstance(data, dict) and 'eeg_data' in data:
            # Extract from JSON request format
            eeg_data = np.array(data['eeg_data'])
        elif not isinstance(data, np.ndarray):
            eeg_data = np.array(data)
        else:
            eeg_data = data
        
        # Handle dimensionality
        if eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(1, -1)
        
        # Use or create stream buffer
        if stream_buffer is not None:
            stream_buffer.add_data(eeg_data)
            
            # Determine adaptive window size based on signal characteristics
            window_size = calculate_adaptive_window(eeg_data)
            eeg_data = stream_buffer.get_window(window_size)
            
            if eeg_data is None or eeg_data.size == 0:
                logger.warning("No data available in buffer")
                return {"error": "No data available", "timestamp": np.datetime64('now').astype(str)}
        
        # Apply artifact cleaning if requested
        if clean_artifacts and eeg_data.shape[1] > 10:  # Only if signal is long enough
            try:
                logger.info("Applying artifact cleaning")
                eeg_data, artifact_report = clean_eeg(eeg_data)
                # Apply EEG preprocessing (filtering)
                for i in range(eeg_data.shape[0]):
                    eeg_data[i] = apply_eeg_preprocessing(eeg_data[i])
            except Exception as e:
                logger.warning(f"Artifact cleaning skipped: {str(e)}")
        
        # Extract features
        logger.info("Extracting features")
        # Create a DataFrame with channels as columns
        import pandas as pd
        df = pd.DataFrame(eeg_data.T, 
                         columns=[f"channel_{i}" for i in range(eeg_data.shape[0])])
        features_df = extract_features(df)
        
        # Preprocess features
        logger.info("Preprocessing data")
        X_processed = preprocess_data(features_df, clean_artifacts=False)
        
        # Load model if not provided (use singleton cache to prevent redundant loading)
        if model is None:
            if model_path is None:
                model_path = "./processed/trained_model.h5"
            
            # Use model cache
            model_cache = ModelCache()
            model = model_cache.get_model(model_path)
        
        # Reshape for CNN input
        X_processed = X_processed.reshape(-1, X_processed.shape[1], 1)
        
        # Get predictions
        logger.info("Getting predictions")
        predictions = model.predict(X_processed, verbose=0)  # Disable verbose output for faster prediction
        predicted_state = np.argmax(predictions, axis=1)
        confidence = np.max(predictions, axis=1) * 100  # Convert to percentage
        
        # Apply temporal smoothing for stability - adapt window size based on data length
        smoothing_window = min(PROCESSING_CONFIG['smoothing_window'], len(predicted_state) // 4)
        if len(predicted_state) > max(5, smoothing_window * 2):  # Only smooth if enough samples
            smoothed_states = temporal_smoothing(predicted_state, smoothing_window)
        else:
            smoothed_states = predicted_state
        
        # Calculate state distribution
        if len(smoothed_states) > 0:
            unique_states, counts = np.unique(smoothed_states, return_counts=True)
            state_distribution = {int(state): int(count) for state, count in zip(unique_states, counts)}
            dominant_state = int(unique_states[np.argmax(counts)])
        else:
            state_distribution = {}
            dominant_state = -1
        
        processing_time = time.time() - start_time
        
        # Return detailed results
        return {
            "predicted_states": [int(s) for s in smoothed_states],
            "dominant_state": dominant_state,
            "confidence": float(np.mean(confidence)),
            "state_distribution": state_distribution,
            "num_samples": len(smoothed_states),
            "processing_time_ms": round(processing_time * 1000, 2),
            "window_size": eeg_data.shape[1] if eeg_data is not None else 0,
            "timestamp": np.datetime64('now').astype(str)
        }
    
    except Exception as e:
        logger.error(f"Real-time processing error: {str(e)}", exc_info=True)
        return {"error": str(e), "timestamp": np.datetime64('now').astype(str)}


# Create a default stream buffer for application-wide use
default_stream_buffer = StreamBuffer(max_size=PROCESSING_CONFIG.get('max_buffer_size', 5000))
