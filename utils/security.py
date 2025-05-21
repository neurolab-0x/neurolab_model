from cryptography.fernet import Fernet
import base64
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DataEncryption:
    """Handles data encryption/decryption"""
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)

    def encrypt_data(self, data: Dict[str, Any]) -> bytes:
        """Encrypt dictionary data"""
        return self.cipher_suite.encrypt(str(data).encode())

    def decrypt_data(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt data back to dictionary"""
        return eval(self.cipher_suite.decrypt(encrypted_data).decode())

def validate_eeg_data(data: Dict[str, Any]) -> bool:
    """Validate EEG data structure"""
    required_fields = ['timestamp', 'features', 'subject_id', 'session_id']
    return all(field in data for field in required_fields)

def sanitize_model_type(model_type: str) -> str:
    """Sanitize model type input"""
    return model_type.lower().strip() 