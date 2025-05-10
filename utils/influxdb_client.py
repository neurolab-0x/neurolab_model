from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
import logging

class InfluxDBManager:
    """Manager class for InfluxDB operations with time-series data."""
    
    def __init__(self, url: str, token: str, org: str, bucket: str):
        """
        Initialize InfluxDB connection.
        
        Parameters:
        -----------
        url : str
            InfluxDB server URL
        token : str
            Authentication token
        org : str
            Organization name
        bucket : str
            Bucket name for data storage
        """
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        self.bucket = bucket
        self.org = org
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def write_eeg_data(self, 
                      data: np.ndarray,
                      channels: List[str],
                      timestamp: datetime,
                      subject_id: str,
                      session_id: str,
                      metadata: Optional[Dict] = None) -> None:
        """
        Write EEG data to InfluxDB.
        
        Parameters:
        -----------
        data : np.ndarray
            EEG data array (channels x timepoints)
        channels : List[str]
            List of channel names
        timestamp : datetime
            Timestamp for the data
        subject_id : str
            Unique identifier for the subject
        session_id : str
            Unique identifier for the session
        metadata : Dict, optional
            Additional metadata to store
        """
        try:
            points = []
            for i, channel in enumerate(channels):
                point = Point("eeg_data")\
                    .tag("subject_id", subject_id)\
                    .tag("session_id", session_id)\
                    .tag("channel", channel)\
                    .time(timestamp, WritePrecision.NS)
                
                # Add channel data
                point.field("amplitude", float(data[i]))
                
                # Add metadata if provided
                if metadata:
                    for key, value in metadata.items():
                        point.field(key, value)
                
                points.append(point)
            
            self.write_api.write(bucket=self.bucket, record=points)
            self.logger.info(f"Successfully wrote EEG data for subject {subject_id}, session {session_id}")
            
        except Exception as e:
            self.logger.error(f"Error writing EEG data: {str(e)}")
            raise
    
    def write_model_predictions(self,
                              predictions: np.ndarray,
                              probabilities: np.ndarray,
                              timestamp: datetime,
                              subject_id: str,
                              session_id: str,
                              model_type: str) -> None:
        """
        Write model predictions to InfluxDB.
        
        Parameters:
        -----------
        predictions : np.ndarray
            Model predictions
        probabilities : np.ndarray
            Prediction probabilities
        timestamp : datetime
            Timestamp for the predictions
        subject_id : str
            Unique identifier for the subject
        session_id : str
            Unique identifier for the session
        model_type : str
            Type of model used for predictions
        """
        try:
            point = Point("model_predictions")\
                .tag("subject_id", subject_id)\
                .tag("session_id", session_id)\
                .tag("model_type", model_type)\
                .time(timestamp, WritePrecision.NS)
            
            # Add prediction data
            point.field("prediction", int(predictions))
            point.field("confidence", float(np.max(probabilities)))
            
            # Add all class probabilities
            for i, prob in enumerate(probabilities):
                point.field(f"class_{i}_probability", float(prob))
            
            self.write_api.write(bucket=self.bucket, record=point)
            self.logger.info(f"Successfully wrote predictions for subject {subject_id}, session {session_id}")
            
        except Exception as e:
            self.logger.error(f"Error writing predictions: {str(e)}")
            raise
    
    def write_personalized_metrics(self,
                                 metrics: Dict[str, float],
                                 timestamp: datetime,
                                 subject_id: str,
                                 session_id: str) -> None:
        """
        Write personalized metrics to InfluxDB.
        
        Parameters:
        -----------
        metrics : Dict[str, float]
            Dictionary of metrics to store
        timestamp : datetime
            Timestamp for the metrics
        subject_id : str
            Unique identifier for the subject
        session_id : str
            Unique identifier for the session
        """
        try:
            point = Point("personalized_metrics")\
                .tag("subject_id", subject_id)\
                .tag("session_id", session_id)\
                .time(timestamp, WritePrecision.NS)
            
            # Add all metrics
            for key, value in metrics.items():
                point.field(key, float(value))
            
            self.write_api.write(bucket=self.bucket, record=point)
            self.logger.info(f"Successfully wrote metrics for subject {subject_id}, session {session_id}")
            
        except Exception as e:
            self.logger.error(f"Error writing metrics: {str(e)}")
            raise
    
    def query_subject_data(self,
                          subject_id: str,
                          start_time: datetime,
                          end_time: datetime,
                          measurement: str = "eeg_data") -> pd.DataFrame:
        """
        Query data for a specific subject within a time range.
        
        Parameters:
        -----------
        subject_id : str
            Unique identifier for the subject
        start_time : datetime
            Start time for the query
        end_time : datetime
            End time for the query
        measurement : str
            Measurement name to query
            
        Returns:
        --------
        pd.DataFrame
            Query results as a pandas DataFrame
        """
        try:
            query = f'''
                from(bucket: "{self.bucket}")
                    |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                    |> filter(fn: (r) => r["_measurement"] == "{measurement}")
                    |> filter(fn: (r) => r["subject_id"] == "{subject_id}")
            '''
            
            result = self.query_api.query_data_frame(query)
            return result
            
        except Exception as e:
            self.logger.error(f"Error querying data: {str(e)}")
            raise
    
    def get_subject_statistics(self,
                             subject_id: str,
                             start_time: datetime,
                             end_time: datetime) -> Dict:
        """
        Get statistical summary for a subject's data.
        
        Parameters:
        -----------
        subject_id : str
            Unique identifier for the subject
        start_time : datetime
            Start time for the analysis
        end_time : datetime
            End time for the analysis
            
        Returns:
        --------
        Dict
            Dictionary containing statistical summaries
        """
        try:
            # Query EEG data
            eeg_data = self.query_subject_data(subject_id, start_time, end_time, "eeg_data")
            
            # Query predictions
            predictions = self.query_subject_data(subject_id, start_time, end_time, "model_predictions")
            
            # Query metrics
            metrics = self.query_subject_data(subject_id, start_time, end_time, "personalized_metrics")
            
            # Calculate statistics
            stats = {
                "eeg_stats": {
                    "mean_amplitude": float(eeg_data["amplitude"].mean()),
                    "std_amplitude": float(eeg_data["amplitude"].std()),
                    "min_amplitude": float(eeg_data["amplitude"].min()),
                    "max_amplitude": float(eeg_data["amplitude"].max())
                },
                "prediction_stats": {
                    "accuracy": float((predictions["prediction"] == predictions["_value"]).mean()),
                    "avg_confidence": float(predictions["confidence"].mean())
                },
                "metrics_stats": {
                    metric: float(metrics[metric].mean())
                    for metric in metrics.columns
                    if metric not in ["_time", "subject_id", "session_id"]
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {str(e)}")
            raise
    
    def get_personalized_insights(self,
                                subject_id: str,
                                start_time: datetime,
                                end_time: datetime) -> Dict:
        """
        Generate personalized insights based on historical data.
        
        Parameters:
        -----------
        subject_id : str
            Unique identifier for the subject
        start_time : datetime
            Start time for the analysis
        end_time : datetime
            End time for the analysis
            
        Returns:
        --------
        Dict
            Dictionary containing personalized insights
        """
        try:
            # Get subject statistics
            stats = self.get_subject_statistics(subject_id, start_time, end_time)
            
            # Generate insights
            insights = {
                "performance_trend": self._analyze_performance_trend(subject_id, start_time, end_time),
                "optimal_conditions": self._identify_optimal_conditions(subject_id, start_time, end_time),
                "recommendations": self._generate_recommendations(stats)
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating insights: {str(e)}")
            raise
    
    def _analyze_performance_trend(self,
                                 subject_id: str,
                                 start_time: datetime,
                                 end_time: datetime) -> Dict:
        """Analyze performance trends over time."""
        try:
            predictions = self.query_subject_data(subject_id, start_time, end_time, "model_predictions")
            
            # Calculate daily averages
            daily_accuracy = predictions.groupby(predictions["_time"].dt.date)["prediction"].mean()
            
            return {
                "trend": "improving" if daily_accuracy.iloc[-1] > daily_accuracy.iloc[0] else "declining",
                "improvement_rate": float((daily_accuracy.iloc[-1] - daily_accuracy.iloc[0]) / len(daily_accuracy))
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance trend: {str(e)}")
            raise
    
    def _identify_optimal_conditions(self,
                                   subject_id: str,
                                   start_time: datetime,
                                   end_time: datetime) -> Dict:
        """Identify optimal conditions for best performance."""
        try:
            predictions = self.query_subject_data(subject_id, start_time, end_time, "model_predictions")
            metrics = self.query_subject_data(subject_id, start_time, end_time, "personalized_metrics")
            
            # Find conditions with highest accuracy
            best_conditions = metrics[predictions["prediction"] == predictions["_value"]].mean()
            
            return {
                metric: float(value)
                for metric, value in best_conditions.items()
                if metric not in ["_time", "subject_id", "session_id"]
            }
            
        except Exception as e:
            self.logger.error(f"Error identifying optimal conditions: {str(e)}")
            raise
    
    def _generate_recommendations(self, stats: Dict) -> List[str]:
        """Generate personalized recommendations based on statistics."""
        recommendations = []
        
        # Analyze EEG statistics
        if stats["eeg_stats"]["std_amplitude"] > 50:  # Example threshold
            recommendations.append("Consider reducing environmental noise during sessions")
        
        # Analyze prediction statistics
        if stats["prediction_stats"]["avg_confidence"] < 0.7:  # Example threshold
            recommendations.append("Practice more sessions to improve model confidence")
        
        # Analyze metrics
        for metric, value in stats["metrics_stats"].items():
            if value < 0.5:  # Example threshold
                recommendations.append(f"Focus on improving {metric.replace('_', ' ')}")
        
        return recommendations
    
    def close(self):
        """Close the InfluxDB connection."""
        self.client.close() 