from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime
import logging
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import ASYNCHRONOUS
from motor.motor_asyncio import AsyncIOMotorClient
from core.config.database import INFLUXDB_CONFIG, MONGODB_CONFIG, SCHEMA_VERSIONS
from core.models.eeg import EEGDataPoint, EEGSession, EEGFeatures
from core.models.events import DetectedEvent

logger = logging.getLogger(__name__)

class DatabaseService:
    """Unified service for database operations"""
    
    def __init__(self):
        """Initialize database connections"""
        self.influx_client = InfluxDBClient(
            url=INFLUXDB_CONFIG['url'],
            token=INFLUXDB_CONFIG['token'],
            org=INFLUXDB_CONFIG['org']
        )
        self.write_api = self.influx_client.write_api(write_options=ASYNCHRONOUS)
        self.query_api = self.influx_client.query_api()
        
        # Initialize MongoDB client
        self.mongo_client = AsyncIOMotorClient(MONGODB_CONFIG['url'])
        self.db = self.mongo_client[MONGODB_CONFIG['database']]
        
    async def store_eeg_data(self, data_point: EEGDataPoint) -> None:
        """Store EEG data in InfluxDB"""
        try:
            point = Point("eeg_measurements")\
                .tag("session_id", data_point.session_id)\
                .tag("subject_id", data_point.subject_id)\
                .tag("schema_version", SCHEMA_VERSIONS['eeg_data'])\
                .time(data_point.timestamp)
                
            # Add all EEG features as fields
            for channel, value in data_point.features.items():
                point = point.field(channel, value)
                
            await self.write_api.write(
                bucket=INFLUXDB_CONFIG['bucket'],
                record=point
            )
        except Exception as e:
            logger.error(f"Error storing EEG data: {str(e)}")
            raise
            
    async def store_session(self, session: EEGSession) -> str:
        """Store session data in MongoDB"""
        try:
            session_data = {
                'session_id': session.session_id,
                'subject_id': session.subject_id,
                'start_time': session.start_time,
                'end_time': session.end_time,
                'duration': session.duration,
                'data_points': [
                    {
                        'timestamp': dp.timestamp,
                        'features': dp.features,
                        'state': dp.state,
                        'confidence': dp.confidence,
                        'metadata': dp.metadata
                    }
                    for dp in session.data_points
                ],
                'metadata': session.metadata,
                'schema_version': SCHEMA_VERSIONS['session_data'],
                'created_at': datetime.utcnow()
            }
            
            result = await self.db[MONGODB_CONFIG['collections']['sessions']].insert_one(session_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error storing session: {str(e)}")
            raise
            
    async def store_event(self, event: DetectedEvent) -> str:
        """Store detected event in MongoDB"""
        try:
            event_data = event.to_dict()
            event_data.update({
                'schema_version': SCHEMA_VERSIONS['event_data'],
                'detected_at': datetime.utcnow()
            })
            
            result = await self.db[MONGODB_CONFIG['collections']['events']].insert_one(event_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error storing event: {str(e)}")
            raise
            
    async def get_session(self, session_id: str) -> Optional[EEGSession]:
        """Retrieve session data from MongoDB"""
        try:
            data = await self.db[MONGODB_CONFIG['collections']['sessions']].find_one(
                {"session_id": session_id}
            )
            
            if not data:
                return None
                
            return EEGSession(
                session_id=data['session_id'],
                subject_id=data['subject_id'],
                start_time=data['start_time'],
                end_time=data['end_time'],
                duration=data['duration'],
                data_points=[
                    EEGDataPoint(
                        timestamp=dp['timestamp'],
                        features=dp['features'],
                        subject_id=data['subject_id'],
                        session_id=data['session_id'],
                        state=dp.get('state'),
                        confidence=dp.get('confidence'),
                        metadata=dp.get('metadata')
                    )
                    for dp in data['data_points']
                ],
                metadata=data['metadata']
            )
        except Exception as e:
            logger.error(f"Error retrieving session: {str(e)}")
            raise
            
    async def get_eeg_data_range(
        self,
        session_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[EEGDataPoint]:
        """Retrieve EEG data from InfluxDB for a specific time range"""
        try:
            query = f'''
                from(bucket: "{INFLUXDB_CONFIG['bucket']}")
                |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                |> filter(fn: (r) => r["session_id"] == "{session_id}")
            '''
            
            result = self.query_api.query(query)
            return [
                EEGDataPoint(
                    timestamp=record.get_time(),
                    features={field: record.get_value() for field in record.get_field()},
                    subject_id=record.values.get('subject_id'),
                    session_id=session_id
                )
                for table in result
                for record in table.records
            ]
        except Exception as e:
            logger.error(f"Error retrieving EEG data: {str(e)}")
            raise
            
    async def get_events_range(
        self,
        session_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[DetectedEvent]:
        """Retrieve events from MongoDB for a specific time range"""
        try:
            cursor = self.db[MONGODB_CONFIG['collections']['events']].find({
                'session_id': session_id,
                'timestamp': {
                    '$gte': start_time,
                    '$lte': end_time
                }
            })
            
            events = []
            async for doc in cursor:
                events.append(DetectedEvent.from_dict(doc))
                
            return events
        except Exception as e:
            logger.error(f"Error retrieving events: {str(e)}")
            raise
            
    async def close(self):
        """Close database connections"""
        try:
            self.influx_client.close()
            self.mongo_client.close()
        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}")
            raise

# Create singleton instance
db_service = DatabaseService() 