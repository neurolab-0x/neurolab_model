import numpy as np
from config.settings import PROCESSING_CONFIG

def temporal_smoothing(states, window_size=None):
    """
    Apply median filtering to state predictions with adaptive window size
    
    Parameters:
    -----------
    states : array-like
        Array of state predictions
    window_size : int or None
        Size of smoothing window, if None uses default from config
    
    Returns:
    --------
    list : Smoothed state predictions
    """
    if window_size is None:
        window_size = PROCESSING_CONFIG['smoothing_window']
    
    # Ensure window size is at least 1
    window_size = max(1, int(window_size))
    
    # Apply median filtering
    return [
        int(np.median(states[max(0,i-window_size):min(len(states),i+window_size+1)]))
        for i in range(len(states))
    ]

def calculate_state_transitions(states):
    """Count state transitions for stability analysis"""
    return sum(1 for i in range(1, len(states)) if states[i] != states[i-1])

def compute_state_stability(states, window_size=10):
    """
    Compute stability score for state predictions
    
    Parameters:
    -----------
    states : array-like
        Array of state predictions
    window_size : int
        Size of window for stability calculation
    
    Returns:
    --------
    float : Stability score (0-100)
        100 = perfectly stable, 0 = highly unstable
    """
    if len(states) < 2:
        return 100.0
    
    # Use sliding windows to calculate local stability
    stability_scores = []
    for i in range(len(states) - window_size + 1):
        window = states[i:i+window_size]
        unique_states = len(set(window))
        # More unique states = less stable
        window_stability = 100 * (1 - (unique_states - 1) / (window_size - 1))
        stability_scores.append(window_stability)
    
    # Return average stability across all windows
    return sum(stability_scores) / len(stability_scores) if stability_scores else 100.0
