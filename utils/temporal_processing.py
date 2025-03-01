import numpy as np
from config.settings import PROCESSING_CONFIG

def temporal_smoothing(states):
    """Apply median filtering to state predictions"""
    window_size = PROCESSING_CONFIG['smoothing_window']
    return [
        np.median(states[max(0,i-window_size):min(len(states),i+window_size)])
        for i in range(len(states))
    ]

def calculate_state_transitions(states):
    """Count state transitions for stability analysis"""
    return sum(1 for i in range(1, len(states)) if states[i] != states[i-1])
