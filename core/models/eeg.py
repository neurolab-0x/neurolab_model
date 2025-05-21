from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List

@dataclass
class EEGDataPoint:
    """Base class for EEG data points"""
    timestamp: datetime
    features: Dict[str, float]
    subject_id: str
    session_id: str
    state: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class EEGSession:
    """Class representing an EEG recording session"""
    session_id: str
    subject_id: str
    start_time: datetime
    end_time: datetime
    data_points: List[EEGDataPoint]
    metadata: Dict[str, Any]
    duration: float = 0.0  # Default value moved to end

    @property
    def duration(self) -> float:
        """Get session duration in seconds"""
        return (self.end_time - self.start_time).total_seconds()

@dataclass
class EEGFeatures:
    """Class representing extracted EEG features"""
    alpha: float
    beta: float
    theta: float
    gamma: float
    delta: float
    signal_quality: float = 1.0  # Default to perfect quality
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EEGFeatures':
        """Create EEGFeatures from a dictionary"""
        return cls(
            alpha=float(data.get('alpha', 0.0)),
            beta=float(data.get('beta', 0.0)),
            theta=float(data.get('theta', 0.0)),
            gamma=float(data.get('gamma', 0.0)),
            delta=float(data.get('delta', 0.0)),
            signal_quality=float(data.get('signal_quality', 1.0)),
            metadata=data.get('metadata')
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'theta': self.theta,
            'gamma': self.gamma,
            'delta': self.delta,
            'signal_quality': self.signal_quality,
            'metadata': self.metadata
        } 