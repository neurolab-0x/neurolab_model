"""Database configuration settings"""

# InfluxDB Configuration
INFLUXDB_CONFIG = {
    'url': 'http://localhost:8086',
    'token': 'your-token-here',  # Replace with actual token
    'org': 'neurolab',
    'bucket': 'eeg_data'
}

# MongoDB Configuration
MONGODB_CONFIG = {
    'url': 'mongodb://localhost:27017',
    'database': 'neurolab',
    'collections': {
        'sessions': 'eeg_sessions',
        'events': 'detected_events',
        'subjects': 'subjects',
        'models': 'ml_models'
    }
}

# Schema Versions
SCHEMA_VERSIONS = {
    'eeg_data': '1.0.0',
    'session_data': '1.0.0',
    'event_data': '1.0.0',
    'subject_data': '1.0.0',
    'model_data': '1.0.0'
}

# Database Indexes
MONGODB_INDEXES = {
    'sessions': [
        {'session_id': 1},
        {'subject_id': 1},
        {'start_time': 1}
    ],
    'events': [
        {'session_id': 1},
        {'timestamp': 1},
        {'event_type': 1}
    ],
    'subjects': [
        {'subject_id': 1}
    ],
    'models': [
        {'model_id': 1},
        {'version': 1}
    ]
}

# InfluxDB Retention Policies
INFLUXDB_RETENTION = {
    'eeg_data': '30d',  # 30 days retention for raw EEG data
    'processed_data': '90d',  # 90 days retention for processed data
    'events': '365d'  # 1 year retention for events
} 