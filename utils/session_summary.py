from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import numpy as np
from utils.database_service import db_service
from utils.event_detector import event_detector, DetectedEvent

logger = logging.getLogger(__name__)

@dataclass
class SessionSummary:
    """Class representing a complete session summary"""
    session_id: str
    subject_id: str
    start_time: datetime
    end_time: datetime
    duration: float  # in seconds
    events: List[DetectedEvent]
    state_distribution: Dict[str, float]
    average_confidence: float
    signal_quality: float
    recommendations: List[str]
    metadata: Dict[str, Any]

class SessionSummaryGenerator:
    """Service for generating comprehensive session summaries"""
    
    def __init__(self):
        """Initialize session summary generator"""
        self.state_weights = {
            'stress': 1.0,
            'relaxation': 1.0,
            'attention': 1.0,
            'anomaly': 2.0  # Higher weight for anomalies
        }
        
    async def generate_summary(
        self,
        session_id: str,
        subject_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> SessionSummary:
        """Generate a comprehensive session summary"""
        try:
            # Retrieve session data
            session_data = await db_service.get_session_data(session_id)
            if not session_data:
                raise ValueError(f"No data found for session {session_id}")
                
            # Retrieve EEG data for the session
            eeg_data = await db_service.get_eeg_data_range(
                session_id,
                start_time,
                end_time
            )
            
            # Calculate session duration
            duration = (end_time - start_time).total_seconds()
            
            # Analyze events
            events = await self._analyze_events(session_id, start_time, end_time)
            
            # Calculate state distribution
            state_distribution = self._calculate_state_distribution(events)
            
            # Calculate average confidence
            average_confidence = self._calculate_average_confidence(events)
            
            # Calculate signal quality
            signal_quality = self._calculate_signal_quality(eeg_data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                events,
                state_distribution,
                signal_quality
            )
            
            # Create summary
            summary = SessionSummary(
                session_id=session_id,
                subject_id=subject_id,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                events=events,
                state_distribution=state_distribution,
                average_confidence=average_confidence,
                signal_quality=signal_quality,
                recommendations=recommendations,
                metadata={
                    'analysis_version': '1.0',
                    'generated_at': datetime.utcnow().isoformat()
                }
            )
            
            # Store summary in database
            await db_service.store_session_summary({
                'session_id': summary.session_id,
                'subject_id': summary.subject_id,
                'start_time': summary.start_time,
                'end_time': summary.end_time,
                'duration': summary.duration,
                'events': [
                    {
                        'event_type': event.event_type,
                        'confidence': event.confidence,
                        'timestamp': event.timestamp,
                        'severity': event.severity,
                        'description': event.description
                    }
                    for event in summary.events
                ],
                'state_distribution': summary.state_distribution,
                'average_confidence': summary.average_confidence,
                'signal_quality': summary.signal_quality,
                'recommendations': summary.recommendations,
                'metadata': summary.metadata
            })
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating session summary: {str(e)}")
            raise
            
    async def _analyze_events(
        self,
        session_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[DetectedEvent]:
        """Analyze events for the session"""
        try:
            # Retrieve events from database
            events = await db_service.get_events_range(
                session_id,
                start_time,
                end_time
            )
            
            # Sort events by timestamp
            events.sort(key=lambda x: x.timestamp)
            
            return events
            
        except Exception as e:
            logger.error(f"Error analyzing events: {str(e)}")
            return []
            
    def _calculate_state_distribution(self, events: List[DetectedEvent]) -> Dict[str, float]:
        """Calculate the distribution of states during the session"""
        try:
            if not events:
                return {}
                
            # Count events by type
            event_counts = {}
            total_weight = 0.0
            
            for event in events:
                weight = self.state_weights.get(event.event_type, 1.0)
                event_counts[event.event_type] = event_counts.get(event.event_type, 0) + weight
                total_weight += weight
                
            # Calculate percentages
            return {
                state: (count / total_weight) * 100
                for state, count in event_counts.items()
            }
            
        except Exception as e:
            logger.error(f"Error calculating state distribution: {str(e)}")
            return {}
            
    def _calculate_average_confidence(self, events: List[DetectedEvent]) -> float:
        """Calculate the average confidence across all events"""
        try:
            if not events:
                return 0.0
                
            confidences = [event.confidence for event in events]
            return sum(confidences) / len(confidences)
            
        except Exception as e:
            logger.error(f"Error calculating average confidence: {str(e)}")
            return 0.0
            
    def _calculate_signal_quality(self, eeg_data: List[Dict[str, Any]]) -> float:
        """Calculate the overall signal quality for the session"""
        try:
            if not eeg_data:
                return 0.0
                
            # Calculate quality for each data point
            qualities = []
            for data_point in eeg_data:
                features = data_point.get('features', {})
                quality = sum(1 for v in features.values() if 0 <= v <= 1) / len(features) if features else 0
                qualities.append(quality)
                
            return sum(qualities) / len(qualities) if qualities else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating signal quality: {str(e)}")
            return 0.0
            
    def _generate_recommendations(
        self,
        events: List[DetectedEvent],
        state_distribution: Dict[str, float],
        signal_quality: float
    ) -> List[str]:
        """Generate recommendations based on session analysis"""
        try:
            recommendations = []
            
            # Check signal quality
            if signal_quality < 0.7:
                recommendations.append(
                    "Consider improving electrode placement for better signal quality"
                )
                
            # Check stress levels
            stress_percentage = state_distribution.get('stress', 0)
            if stress_percentage > 30:
                recommendations.append(
                    "High stress levels detected. Consider stress management techniques"
                )
                
            # Check relaxation levels
            relaxation_percentage = state_distribution.get('relaxation', 0)
            if relaxation_percentage < 20:
                recommendations.append(
                    "Low relaxation levels detected. Consider relaxation exercises"
                )
                
            # Check for anomalies
            if any(event.event_type == 'anomaly' for event in events):
                recommendations.append(
                    "Unusual patterns detected. Consider consulting a specialist"
                )
                
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []

# Create singleton instance
session_summary_generator = SessionSummaryGenerator() 