import logging
from typing import Dict, Any, Optional
from core.ml.processing import (
    load_data,
    label_eeg_states,
    extract_features,
    preprocess_data,
    temporal_smoothing,
    calculate_state_durations,
    generate_recommendations
)

logger = logging.getLogger(__name__)

class MLProcessor:
    def __init__(self):
        self.functions = {
            'load_data': load_data,
            'label_eeg_states': label_eeg_states,
            'extract_features': extract_features,
            'preprocess_data': preprocess_data,
            'temporal_smoothing': temporal_smoothing,
            'calculate_state_durations': calculate_state_durations,
            'generate_recommendations': generate_recommendations
        }
        logger.info("ML processing modules loaded successfully")
    
    def process_eeg_data(self, file_path: str, subject_id: str, session_id: str) -> Dict[str, Any]:
        """Process EEG data through the complete pipeline"""
        try:
            # Load and preprocess data
            data = self.functions['load_data'](file_path)
            features = self.functions['extract_features'](data)
            processed_data = self.functions['preprocess_data'](features)
            
            # Analyze states
            states = self.functions['label_eeg_states'](processed_data)
            smoothed_states = self.functions['temporal_smoothing'](states)
            durations = self.functions['calculate_state_durations'](smoothed_states)
            
            # Generate recommendations
            recommendations = self.functions['generate_recommendations'](
                smoothed_states,
                durations,
                subject_id=subject_id,
                session_id=session_id
            )
            
            return {
                'states': smoothed_states,
                'durations': durations,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error processing EEG data: {str(e)}")
            raise 