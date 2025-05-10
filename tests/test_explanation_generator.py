import unittest
from datetime import datetime
from utils.explanation_generator import ExplanationGenerator, EEGState

class TestExplanationGenerator(unittest.TestCase):
    def setUp(self):
        self.explainer = ExplanationGenerator()
        
    def test_stress_case_scenario(self):
        # Create a realistic EEG state for a stressed patient
        eeg_state = EEGState(
            state="stressed",
            confidence=0.92,
            features={
                "beta": 25.5,  # Elevated beta waves
                "alpha": 7.2,  # Reduced alpha waves
                "theta": 3.8,  # Low theta waves
                "beta_amplitude": 35.0,  # Exceeds safety threshold
                "alpha_amplitude": 45.0,
                "theta_amplitude": 30.0
            },
            timestamp=datetime.now(),
            subject_id="SUBJ001",
            session_id="SESS003"
        )
        
        # Additional context about the patient
        additional_context = {
            "patient_age": 35,
            "occupation": "Software Engineer",
            "session_duration": "45 minutes",
            "previous_sessions": 2,
            "reported_symptoms": "Difficulty concentrating, increased heart rate",
            "medication": "None",
            "sleep_hours": "6 hours"
        }
        
        # Generate explanation
        explanation = self.explainer.generate_explanation(eeg_state, additional_context)
        formatted_report = self.explainer.format_explanation(explanation)
        
        # Print the report for demonstration
        print("\n=== Real-World Case Scenario Report ===\n")
        print(formatted_report)
        
        # Basic assertions to ensure report structure
        self.assertIn("Clinical Observation", formatted_report)
        self.assertIn("Technical Analysis", formatted_report)
        self.assertIn("Interpretation", formatted_report)
        self.assertIn("Recommendations", formatted_report)
        self.assertIn("Additional Context", formatted_report)
        
        # Verify safety thresholds are properly handled
        self.assertIn("exceeds safety limits", formatted_report)
        
        # Verify patient context is included
        self.assertIn("Software Engineer", formatted_report)
        self.assertIn("35", formatted_report)

if __name__ == '__main__':
    unittest.main() 