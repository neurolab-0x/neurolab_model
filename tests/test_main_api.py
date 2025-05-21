import unittest
import os
import json
import numpy as np
from datetime import datetime
from fastapi.testclient import TestClient
from main import app
import pandas as pd

class TestMainAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.test_data_dir = "test_data"
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create sample EEG data
        self.sample_data = pd.DataFrame({
            'timestamp': [datetime.now().isoformat() for _ in range(100)],
            'channel_1': np.random.randn(100),
            'channel_2': np.random.randn(100),
            'channel_3': np.random.randn(100),
            'label': np.random.randint(0, 3, 100)
        })
        
        # Save sample data
        self.csv_path = os.path.join(self.test_data_dir, "test_eeg.csv")
        self.sample_data.to_csv(self.csv_path, index=False)
        
        # Create test user token
        self.test_token = "test_token"  # In real tests, generate proper JWT token
        
    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        if os.path.exists(self.test_data_dir):
            os.rmdir(self.test_data_dir)
            
    def test_health_check(self):
        """Test the health check endpoint"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("model_loaded", data)
        
    def test_upload_endpoint(self):
        """Test the file upload and processing endpoint"""
        with open(self.csv_path, "rb") as f:
            response = self.client.post(
                "/upload",
                files={"file": ("test_eeg.csv", f, "text/csv")},
                headers={"Authorization": f"Bearer {self.test_token}"}
            )
            
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("temporal_analysis", data)
        self.assertIn("cognitive_metrics", data)
        self.assertIn("clinical_recommendations", data)
        self.assertIn("medical_explanations", data)
        
    def test_realtime_endpoint(self):
        """Test the real-time data processing endpoint"""
        test_data = {
            "features": {
                "channel_1": 0.5,
                "channel_2": -0.3,
                "channel_3": 0.1
            },
            "session_id": "test_session_001"
        }
        
        response = self.client.post(
            "/realtime/",
            json=test_data,
            headers={"Authorization": f"Bearer {self.test_token}"}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("explanation", data)
        self.assertIn("timestamp", data)
        
    def test_interpretability_endpoint(self):
        """Test the model interpretability endpoint"""
        with open(self.csv_path, "rb") as f:
            response = self.client.post(
                "/interpretability/explain",
                files={"file": ("test_eeg.csv", f, "text/csv")},
                params={"explanation_type": "shap", "num_samples": 5},
                headers={"Authorization": f"Bearer {self.test_token}"}
            )
            
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("explanation_type", data)
        self.assertIn("results", data)
        self.assertIn("model_info", data)
        
    def test_invalid_file_upload(self):
        """Test handling of invalid file uploads"""
        # Test with empty file
        empty_file = os.path.join(self.test_data_dir, "empty.csv")
        with open(empty_file, "w") as f:
            pass
            
        with open(empty_file, "rb") as f:
            response = self.client.post(
                "/upload",
                files={"file": ("empty.csv", f, "text/csv")},
                headers={"Authorization": f"Bearer {self.test_token}"}
            )
            
        self.assertEqual(response.status_code, 400)
        os.remove(empty_file)
        
        # Test with unsupported file type
        invalid_file = os.path.join(self.test_data_dir, "test.txt")
        with open(invalid_file, "w") as f:
            f.write("Invalid data")
            
        with open(invalid_file, "rb") as f:
            response = self.client.post(
                "/upload",
                files={"file": ("test.txt", f, "text/plain")},
                headers={"Authorization": f"Bearer {self.test_token}"}
            )
            
        self.assertEqual(response.status_code, 400)
        os.remove(invalid_file)
        
    def test_realtime_invalid_data(self):
        """Test handling of invalid real-time data"""
        # Test with missing features
        invalid_data = {
            "session_id": "test_session_001"
        }
        
        response = self.client.post(
            "/realtime/",
            json=invalid_data,
            headers={"Authorization": f"Bearer {self.test_token}"}
        )
        
        self.assertEqual(response.status_code, 500)
        
    def test_interpretability_invalid_params(self):
        """Test handling of invalid interpretability parameters"""
        with open(self.csv_path, "rb") as f:
            response = self.client.post(
                "/interpretability/explain",
                files={"file": ("test_eeg.csv", f, "text/csv")},
                params={"explanation_type": "invalid_type", "num_samples": 5},
                headers={"Authorization": f"Bearer {self.test_token}"}
            )
            
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main() 