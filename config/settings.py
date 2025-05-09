PROCESSING_CONFIG = {
    'sample_rate': 0.5,
    'smoothing_window': 5,
    'max_timeline_points': 100,
    # Streaming buffer settings
    'max_buffer_size': 5000,  # Maximum samples to retain in buffer
    'min_window_size': 50,    # Minimum window size for adaptive processing
    'max_window_size': 500,   # Maximum window size for adaptive processing
    'buffer_overlap_percent': 25  # Overlap percentage between consecutive windows
}

THRESHOLDS = {
    'stress_threshold': 0.25,
    'confidence_threshold': 70,
    'severe_stress_threshold': 0.4,
    'relaxation_threshold': 0.6,
    'max_recommendations': 3
}

REAL_TIME_CONFIG = {
    'enable_caching': True,         # Enable model caching
    'enable_streaming': True,       # Enable streaming buffer
    'enable_adaptive_window': True, # Enable adaptive window sizing
    'latency_threshold_ms': 100     # Target latency threshold in milliseconds
}

SECURITY_CONFIG = {
    # Encryption settings
    'enable_encryption': True,              # Enable data encryption
    'encryption_for_storage': True,         # Encrypt data before storing
    'encryption_for_transit': True,         # Always encrypt responses
    
    # Authentication settings
    'require_authentication': True,         # Require authentication for API access
    'token_expiry_hours': 24,               # JWT token expiry time in hours
    'refresh_token_expiry_days': 30,        # Refresh token expiry in days
    
    # Authorization settings
    'required_role_realtime': 'user',       # Role required for real-time processing
    'required_role_admin': 'admin',         # Role required for administrative functions
    
    # Rate limiting
    'enable_rate_limiting': True,           # Enable rate limiting
    'rate_limit_requests': 60,              # Maximum requests per window
    'rate_limit_window_seconds': 60,        # Window size in seconds
    
    # Data validation
    'max_eeg_amplitude': 1000,              # Maximum allowed amplitude in microvolts
    'max_eeg_channels': 64,                 # Maximum allowed EEG channels
    'max_eeg_samples': 10000                # Maximum samples per request
}

# Logging configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_security_events': True,
    'security_log_path': './logs/security.log',
    'application_log_path': './logs/application.log'
}

MODEL_VERSION = "nlPT 1-Preview"
MODEL_NAME = "neurai nlPT"
