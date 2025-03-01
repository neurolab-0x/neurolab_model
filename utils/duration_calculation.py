from collections import defaultdict
from config.settings import PROCESSING_CONFIG

def calculate_state_durations(states):
    """Calculate time spent in each cognitive state"""
    durations = defaultdict(float)
    if not states:
        return durations
    
    current_state = states[0]
    start_idx = 0
    sample_rate = PROCESSING_CONFIG['sample_rate']
    
    for i, state in enumerate(states):
        if state != current_state:
            duration = (i - start_idx) * sample_rate
            durations[current_state] += duration
            current_state = state
            start_idx = i
    
    # Add final state duration
    duration = (len(states) - start_idx) * sample_rate
    durations[current_state] += duration
    
    return durations
