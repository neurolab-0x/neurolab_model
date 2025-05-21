from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime
import logging
from dataclasses import dataclass
from utils.database_service import db_service

logger = logging.getLogger(__name__)

@dataclass
class DetectedEvent:
    """Class representing a detected neurological event"""
    event_type: str
    confidence: float
    timestamp: datetime
    session_id: str
    subject_id: str
    features: Dict[str, float]
    severity: str
    description: str
    metadata: Dict[str, Any]

class EventDetector:
    """Service for detecting neurological events and patterns"""
    
    def __init__(self):
        """Initialize event detector with thresholds and patterns"""
        self.patterns = {
            'stress': {
                'threshold': 0.7,
                'min_duration': 5,  # seconds
                'required_channels': ['beta', 'alpha']
            },
            'relaxation': {
                'threshold': 0.6,
                'min_duration': 10,  # seconds
                'required_channels': ['theta', 'alpha']
            },
            'attention': {
                'threshold': 0.8,
                'min_duration': 3,  # seconds
                'required_channels': ['beta', 'gamma']
            }
        }
        
        self.anomaly_detector = self._initialize_anomaly_detector()
        
    def _initialize_anomaly_detector(self):
        """Initialize anomaly detection model"""
        # TODO: Implement more sophisticated anomaly detection
        return None
        
    async def detect_events(
        self,
        data: Dict[str, Any],
        session_id: str,
        subject_id: str
    ) -> List[DetectedEvent]:
        """Detect events in the current data window"""
        try:
            events = []
            
            # Check for known patterns
            for pattern_name, pattern_config in self.patterns.items():
                if self._check_pattern(data, pattern_config):
                    event = DetectedEvent(
                        event_type=pattern_name,
                        confidence=self._calculate_confidence(data, pattern_config),
                        timestamp=datetime.utcnow(),
                        session_id=session_id,
                        subject_id=subject_id,
                        features=data['features'],
                        severity=self._determine_severity(data, pattern_name),
                        description=self._generate_description(pattern_name, data),
                        metadata={'pattern_config': pattern_config}
                    )
                    events.append(event)
                    
                    # Store event in database
                    await db_service.store_detected_event({
                        'event_type': event.event_type,
                        'confidence': event.confidence,
                        'timestamp': event.timestamp,
                        'session_id': event.session_id,
                        'subject_id': event.subject_id,
                        'features': event.features,
                        'severity': event.severity,
                        'description': event.description,
                        'metadata': event.metadata
                    })
            
            # Check for anomalies
            if self._detect_anomaly(data):
                anomaly_event = DetectedEvent(
                    event_type='anomaly',
                    confidence=self._calculate_anomaly_confidence(data),
                    timestamp=datetime.utcnow(),
                    session_id=session_id,
                    subject_id=subject_id,
                    features=data['features'],
                    severity='high',
                    description='Detected unusual pattern in EEG data',
                    metadata={'anomaly_type': 'unknown'}
                )
                events.append(anomaly_event)
                
                # Store anomaly event
                await db_service.store_detected_event({
                    'event_type': anomaly_event.event_type,
                    'confidence': anomaly_event.confidence,
                    'timestamp': anomaly_event.timestamp,
                    'session_id': anomaly_event.session_id,
                    'subject_id': anomaly_event.subject_id,
                    'features': anomaly_event.features,
                    'severity': anomaly_event.severity,
                    'description': anomaly_event.description,
                    'metadata': anomaly_event.metadata
                })
            
            return events
            
        except Exception as e:
            logger.error(f"Error detecting events: {str(e)}")
            raise
            
    def _check_pattern(self, data: Dict[str, Any], pattern_config: Dict[str, Any]) -> bool:
        """Check if data matches a specific pattern"""
        try:
            features = data['features']
            
            # Check if all required channels are present
            if not all(channel in features for channel in pattern_config['required_channels']):
                return False
                
            # Calculate pattern score
            score = self._calculate_pattern_score(features, pattern_config)
            
            return score >= pattern_config['threshold']
            
        except Exception as e:
            logger.error(f"Error checking pattern: {str(e)}")
            return False
            
    def _calculate_pattern_score(
        self,
        features: Dict[str, float],
        pattern_config: Dict[str, Any]
    ) -> float:
        """Calculate how well the data matches a pattern"""
        try:
            # Simple weighted average of relevant features
            weights = {
                'beta': 0.4,
                'alpha': 0.3,
                'theta': 0.2,
                'gamma': 0.1
            }
            
            score = 0.0
            total_weight = 0.0
            
            for channel in pattern_config['required_channels']:
                if channel in features:
                    score += features[channel] * weights.get(channel, 0.1)
                    total_weight += weights.get(channel, 0.1)
                    
            return score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating pattern score: {str(e)}")
            return 0.0
            
    def _calculate_confidence(
        self,
        data: Dict[str, Any],
        pattern_config: Dict[str, Any]
    ) -> float:
        """Calculate confidence in the detected pattern"""
        try:
            base_confidence = self._calculate_pattern_score(data['features'], pattern_config)
            
            # Adjust confidence based on signal quality
            signal_quality = self._assess_signal_quality(data['features'])
            
            return base_confidence * signal_quality
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0
            
    def _determine_severity(self, data: Dict[str, Any], pattern_name: str) -> str:
        """Determine the severity of a detected event"""
        try:
            if pattern_name == 'stress':
                if data['features'].get('beta', 0) > 0.8:
                    return 'high'
                elif data['features'].get('beta', 0) > 0.6:
                    return 'medium'
                return 'low'
                
            elif pattern_name == 'relaxation':
                if data['features'].get('alpha', 0) > 0.8:
                    return 'high'
                elif data['features'].get('alpha', 0) > 0.6:
                    return 'medium'
                return 'low'
                
            return 'medium'
            
        except Exception as e:
            logger.error(f"Error determining severity: {str(e)}")
            return 'low'
            
    def _generate_description(self, pattern_name: str, data: Dict[str, Any]) -> str:
        """Generate a human-readable description of the detected event"""
        try:
            if pattern_name == 'stress':
                return f"Detected stress pattern with beta activity at {data['features'].get('beta', 0):.2f}"
            elif pattern_name == 'relaxation':
                return f"Detected relaxation pattern with alpha activity at {data['features'].get('alpha', 0):.2f}"
            elif pattern_name == 'attention':
                return f"Detected attention pattern with beta/gamma activity at {data['features'].get('beta', 0):.2f}"
            return f"Detected {pattern_name} pattern"
            
        except Exception as e:
            logger.error(f"Error generating description: {str(e)}")
            return f"Detected {pattern_name} pattern"
            
    def _detect_anomaly(self, data: Dict[str, Any]) -> bool:
        """Detect anomalies in the data"""
        # TODO: Implement more sophisticated anomaly detection
        return False
        
    def _calculate_anomaly_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence in detected anomaly"""
        # TODO: Implement anomaly confidence calculation
        return 0.0
        
    def _assess_signal_quality(self, features: Dict[str, float]) -> float:
        """Assess the quality of the EEG signal"""
        try:
            # Simple signal quality assessment
            quality_indicators = []
            
            # Check for reasonable ranges
            for channel, value in features.items():
                if 0 <= value <= 1:
                    quality_indicators.append(1.0)
                else:
                    quality_indicators.append(0.0)
                    
            return sum(quality_indicators) / len(quality_indicators) if quality_indicators else 0.0
            
        except Exception as e:
            logger.error(f"Error assessing signal quality: {str(e)}")
            return 0.0

# Create singleton instance
event_detector = EventDetector() 