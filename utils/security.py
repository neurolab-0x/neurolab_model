import os
import base64
import hashlib
import hmac
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt
import re
import numpy as np

logger = logging.getLogger(__name__)

# ========== DATA ENCRYPTION ==========

class DataEncryption:
    """Handles encryption and decryption of sensitive EEG data"""
    
    def __init__(self, encryption_key=None):
        """Initialize with encryption key or generate one"""
        if encryption_key:
            self.key = encryption_key
        else:
            # Use environment variable or generate a key
            self.key = os.environ.get('EEG_ENCRYPTION_KEY', None)
            if not self.key:
                self.key = self._generate_key()
        
        # Initialize Fernet cipher with the key
        try:
            # Convert string key to bytes if necessary
            if isinstance(self.key, str):
                self.key = self.key.encode()
                
            # Ensure key is proper Fernet key (32 url-safe base64-encoded bytes)
            if len(self.key) != 32 or not all(c in b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_=' for c in self.key):
                # Derive a proper key using PBKDF2
                salt = b'eeg_neurolab_salt'  # This should be stored securely in production
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                self.key = base64.urlsafe_b64encode(kdf.derive(self.key))
            
            self.cipher = Fernet(self.key)
            logger.info("Encryption initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing encryption: {str(e)}")
            raise
    
    def _generate_key(self) -> bytes:
        """Generate a new encryption key"""
        key = Fernet.generate_key()
        logger.warning("Generated new encryption key. This should be stored securely.")
        return key
    
    def encrypt_data(self, data: Any) -> bytes:
        """Encrypt any data that can be serialized to JSON"""
        try:
            import json
            serialized = json.dumps(data)
            encrypted = self.cipher.encrypt(serialized.encode())
            return encrypted
        except Exception as e:
            logger.error(f"Encryption error: {str(e)}")
            raise
    
    def decrypt_data(self, encrypted_data: bytes) -> Any:
        """Decrypt data and return the original object"""
        try:
            import json
            decrypted = self.cipher.decrypt(encrypted_data)
            return json.loads(decrypted.decode())
        except Exception as e:
            logger.error(f"Decryption error: {str(e)}")
            raise
    
    def encrypt_eeg_data(self, eeg_data: np.ndarray) -> Dict[str, Any]:
        """Specifically encrypt EEG data with metadata"""
        try:
            # Convert numpy array to list for serialization
            data_list = eeg_data.tolist()
            
            # Create metadata
            metadata = {
                "shape": eeg_data.shape,
                "dtype": str(eeg_data.dtype),
                "timestamp": datetime.now().isoformat(),
                "encrypted": True
            }
            
            # Create combined payload
            payload = {
                "metadata": metadata,
                "data": data_list
            }
            
            # Encrypt the payload
            encrypted = self.encrypt_data(payload)
            
            return {
                "encrypted_data": base64.b64encode(encrypted).decode(),
                "metadata": {
                    "shape": eeg_data.shape,
                    "timestamp": metadata["timestamp"],
                    "encrypted": True
                }
            }
        except Exception as e:
            logger.error(f"EEG encryption error: {str(e)}")
            raise
    
    def decrypt_eeg_data(self, encrypted_payload: Dict[str, Any]) -> np.ndarray:
        """Decrypt EEG data from an encrypted payload"""
        try:
            # Extract and decode the encrypted data
            encrypted_data = base64.b64decode(encrypted_payload["encrypted_data"])
            
            # Decrypt the payload
            payload = self.decrypt_data(encrypted_data)
            
            # Extract the data and metadata
            data_list = payload["data"]
            metadata = payload["metadata"]
            
            # Reconstruct numpy array
            eeg_data = np.array(data_list, dtype=metadata["dtype"])
            
            return eeg_data
        except Exception as e:
            logger.error(f"EEG decryption error: {str(e)}")
            raise


# ========== AUTHENTICATION & AUTHORIZATION ==========

class JWTAuth:
    """JWT-based authentication and authorization"""
    
    def __init__(self, secret_key=None, algorithm='HS256', token_expiry=24):
        """Initialize with secret key or generate one"""
        if secret_key:
            self.secret_key = secret_key
        else:
            # Use environment variable or generate a key
            self.secret_key = os.environ.get('JWT_SECRET_KEY', None)
            if not self.secret_key:
                self.secret_key = self._generate_secret()
        
        self.algorithm = algorithm
        self.token_expiry = token_expiry  # in hours
    
    def _generate_secret(self) -> str:
        """Generate a new secret key"""
        secret = base64.b64encode(os.urandom(32)).decode()
        logger.warning("Generated new JWT secret key. This should be stored securely.")
        return secret
    
    def create_token(self, user_id: str, roles: List[str] = None) -> str:
        """Create a JWT token for a user"""
        try:
            roles = roles or ["user"]
            
            # Set expiration time
            expiry = datetime.utcnow() + timedelta(hours=self.token_expiry)
            
            # Create the payload
            payload = {
                "sub": user_id,
                "roles": roles,
                "exp": expiry
            }
            
            # Create the token
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            
            return token
        except Exception as e:
            logger.error(f"Token creation error: {str(e)}")
            raise
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify a JWT token and return the payload"""
        try:
            # Decode and verify the token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            raise ValueError("Token expired")
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            raise ValueError("Invalid token")
        except Exception as e:
            logger.error(f"Token verification error: {str(e)}")
            raise
    
    def has_role(self, token: str, required_role: str) -> bool:
        """Check if a token has a specific role"""
        try:
            payload = self.verify_token(token)
            roles = payload.get("roles", [])
            
            return required_role in roles
        except Exception:
            return False


# ========== INPUT VALIDATION & SANITIZATION ==========

def validate_eeg_data(eeg_data) -> bool:
    """
    Validate EEG data structure and values
    
    Returns:
    --------
    bool: True if valid, False otherwise
    """
    try:
        # Convert to numpy array if not already
        if not isinstance(eeg_data, np.ndarray):
            eeg_data = np.array(eeg_data)
        
        # Check dimensionality (should be 1D or 2D)
        if eeg_data.ndim not in [1, 2]:
            return False
        
        # Check for NaN or Inf values
        if np.isnan(eeg_data).any() or np.isinf(eeg_data).any():
            return False
        
        # Check for reasonable amplitude values (typical EEG is -100 to 100 microvolts)
        if np.abs(eeg_data).max() > 1000:  # Allow slightly higher than typical for flexibility
            return False
        
        return True
    except Exception as e:
        logger.error(f"EEG validation error: {str(e)}")
        return False

def sanitize_model_type(model_type: str) -> str:
    """
    Sanitize model type string to prevent path traversal and injection
    
    Returns:
    --------
    str: Sanitized model type
    """
    if not model_type:
        return "original"
    
    # Remove any path traversal attempts
    model_type = os.path.basename(model_type)
    
    # Allow only alphanumeric, underscore, and hyphen
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', model_type)
    
    # If anything was filtered out, use default
    if sanitized != model_type:
        logger.warning(f"Model type sanitized from '{model_type}' to '{sanitized}'")
    
    return sanitized

def validate_client_id(client_id: str) -> bool:
    """
    Validate client ID format
    
    Returns:
    --------
    bool: True if valid, False otherwise
    """
    if not client_id:
        return False
    
    # Ensure reasonable length
    if len(client_id) > 64:
        return False
    
    # Allow only alphanumeric, underscore, hyphen, and dot
    if not re.match(r'^[a-zA-Z0-9_.-]+$', client_id):
        return False
    
    return True

def generate_rate_limit_key(client_id: str, endpoint: str) -> str:
    """Generate a key for rate limiting"""
    return f"ratelimit:{client_id}:{endpoint}"

def check_rate_limit(client_id: str, endpoint: str, max_requests: int = 60, window_seconds: int = 60) -> bool:
    """
    Check if a client has exceeded the rate limit
    
    Returns:
    --------
    bool: True if within limit, False if exceeded
    """
    # This is a simple implementation
    # In production, you would use Redis or a similar cache
    
    current_time = int(time.time())
    key = generate_rate_limit_key(client_id, endpoint)
    
    # Get existing requests in the current window
    # This is simplified - in production use Redis ZSET with timestamps
    
    # For now, always return True (not rate limited)
    # This is just a placeholder for the actual implementation
    return True 