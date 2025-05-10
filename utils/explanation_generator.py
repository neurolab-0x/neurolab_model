from typing import Dict, List, Union, Optional
import numpy as np
from datetime import datetime
import logging
from transformers import pipeline
import torch
from dataclasses import dataclass
import json
import os

@dataclass
class EEGState:
    """Class to represent EEG state and its characteristics."""
    state: str
    confidence: float
    features: Dict[str, float]
    timestamp: datetime
    subject_id: str
    session_id: str

class ExplanationGenerator:
    """Generates professional medical explanations for EEG analysis results."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the explanation generator.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to a custom trained model for explanation generation
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load the explanation generation model
        try:
            if model_path and os.path.exists(model_path):
                self.explainer = pipeline(
                    "text-generation",
                    model=model_path,
                    device=0 if torch.cuda.is_available() else -1
                )
            else:
                # Use a pre-trained model for medical text generation
                self.explainer = pipeline(
                    "text-generation",
                    model="gpt2",  # Replace with a medical-specific model if available
                    device=0 if torch.cuda.is_available() else -1
                )
        except Exception as e:
            self.logger.error(f"Error loading explanation model: {str(e)}")
            raise
        
        # Load medical terminology and guidelines
        self.medical_terms = self._load_medical_terminology()
        self.guidelines = self._load_medical_guidelines()
    
    def _load_medical_terminology(self) -> Dict:
        """Load medical terminology for EEG analysis."""
        return {
            "calm": {
                "terms": ["alpha waves", "relaxed state", "mental calmness"],
                "normal_range": {"alpha": (8, 13), "theta": (4, 8)},
                "clinical_significance": "Indicates a relaxed, alert state of mind",
                "occupation_specific": {
                    "software_engineer": "Optimal state for complex problem-solving",
                    "healthcare_worker": "Good state for patient care and decision-making",
                    "student": "Ideal state for learning and information retention"
                }
            },
            "focused": {
                "terms": ["beta waves", "concentration", "mental focus"],
                "normal_range": {"beta": (13, 30)},
                "clinical_significance": "Suggests active mental engagement and concentration",
                "occupation_specific": {
                    "software_engineer": "Good for coding and debugging tasks",
                    "healthcare_worker": "Appropriate for medical procedures",
                    "student": "Suitable for studying and exam preparation"
                }
            },
            "drowsy": {
                "terms": ["theta waves", "drowsiness", "light sleep"],
                "normal_range": {"theta": (4, 8)},
                "clinical_significance": "May indicate transition to sleep or reduced alertness",
                "occupation_specific": {
                    "software_engineer": "Risk for code errors and reduced productivity",
                    "healthcare_worker": "Safety concern for patient care",
                    "student": "May affect learning and retention"
                }
            },
            "stressed": {
                "terms": ["high beta", "stress response", "mental tension"],
                "normal_range": {"beta": (20, 30)},
                "clinical_significance": "Suggests heightened mental activity and potential stress",
                "occupation_specific": {
                    "software_engineer": "May impact code quality and team collaboration",
                    "healthcare_worker": "Could affect patient care quality",
                    "student": "May hinder learning and performance"
                }
            }
        }
    
    def _load_medical_guidelines(self) -> Dict:
        """Load medical guidelines for EEG interpretation."""
        return {
            "explanation_format": {
                "sections": [
                    "Clinical Observation",
                    "Technical Analysis",
                    "Interpretation",
                    "Recommendations",
                    "Temporal Analysis",
                    "Safety Assessment"
                ],
                "required_elements": [
                    "wave patterns",
                    "frequency bands",
                    "clinical context",
                    "safety considerations",
                    "temporal changes",
                    "occupation-specific factors"
                ]
            },
            "safety_thresholds": {
                "alpha_amplitude": {
                    "default": 50,
                    "software_engineer": 45,
                    "healthcare_worker": 40,
                    "student": 55
                },
                "beta_amplitude": {
                    "default": 30,
                    "software_engineer": 35,
                    "healthcare_worker": 25,
                    "student": 32
                },
                "theta_amplitude": {
                    "default": 40,
                    "software_engineer": 35,
                    "healthcare_worker": 30,
                    "student": 45
                }
            },
            "alert_levels": {
                "mild": {
                    "threshold": 0.7,
                    "action": "Monitor and document"
                },
                "moderate": {
                    "threshold": 0.85,
                    "action": "Immediate intervention recommended"
                },
                "severe": {
                    "threshold": 0.95,
                    "action": "Urgent medical attention required"
                }
            }
        }
    
    def generate_explanation(self, 
                           eeg_state: EEGState,
                           additional_context: Optional[Dict] = None) -> Dict:
        """
        Generate a professional medical explanation for the EEG state.
        
        Parameters:
        -----------
        eeg_state : EEGState
            The detected EEG state and its characteristics
        additional_context : Dict, optional
            Additional context for the explanation
            
        Returns:
        --------
        Dict
            Structured explanation with different sections
        """
        try:
            # Get state-specific terminology
            state_info = self.medical_terms.get(eeg_state.state.lower(), {})
            
            # Generate safety assessment
            safety_assessment = self._generate_safety_assessment(eeg_state, additional_context)
            
            # Generate the explanation structure
            explanation = {
                "clinical_observation": self._generate_clinical_observation(eeg_state, state_info),
                "technical_analysis": self._generate_technical_analysis(eeg_state, state_info),
                "interpretation": self._generate_interpretation(eeg_state, state_info),
                "recommendations": self._generate_recommendations(eeg_state, state_info),
                "temporal_analysis": self._generate_temporal_analysis(eeg_state, additional_context),
                "safety_assessment": safety_assessment,
                "metadata": {
                    "timestamp": eeg_state.timestamp.isoformat(),
                    "subject_id": eeg_state.subject_id,
                    "session_id": eeg_state.session_id,
                    "confidence": float(eeg_state.confidence)
                }
            }
            
            # Add additional context if provided
            if additional_context:
                explanation["additional_context"] = additional_context
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error generating explanation: {str(e)}")
            raise
    
    def _generate_clinical_observation(self, 
                                     eeg_state: EEGState,
                                     state_info: Dict) -> str:
        """Generate the clinical observation section."""
        try:
            # Get relevant medical terms
            terms = state_info.get("terms", [])
            
            # Generate the observation text
            observation = f"Clinical observation indicates a state of {eeg_state.state.lower()} "
            observation += f"with {len(terms)} key characteristics: {', '.join(terms)}. "
            observation += f"The confidence level of this assessment is {eeg_state.confidence:.2%}. "
            observation += state_info.get("clinical_significance", "")
            
            return observation
            
        except Exception as e:
            self.logger.error(f"Error generating clinical observation: {str(e)}")
            raise
    
    def _generate_technical_analysis(self,
                                   eeg_state: EEGState,
                                   state_info: Dict) -> str:
        """Generate the technical analysis section."""
        try:
            analysis = "Technical Analysis:\n"
            
            # Add wave pattern analysis
            analysis += "Wave Patterns:\n"
            for feature, value in eeg_state.features.items():
                if feature in state_info.get("normal_range", {}):
                    normal_range = state_info["normal_range"][feature]
                    analysis += f"- {feature}: {value:.2f} Hz "
                    if normal_range[0] <= value <= normal_range[1]:
                        analysis += "(within normal range)\n"
                    else:
                        analysis += "(outside normal range)\n"
            
            # Add amplitude analysis
            analysis += "\nAmplitude Analysis:\n"
            for feature, value in eeg_state.features.items():
                if "amplitude" in feature.lower():
                    threshold = self.guidelines["safety_thresholds"].get(feature, 0)
                    analysis += f"- {feature}: {value:.2f} μV "
                    if value <= threshold:
                        analysis += "(within safety limits)\n"
                    else:
                        analysis += "(exceeds safety limits)\n"
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error generating technical analysis: {str(e)}")
            raise
    
    def _generate_interpretation(self,
                               eeg_state: EEGState,
                               state_info: Dict) -> str:
        """Generate the interpretation section."""
        try:
            # Use the explainer model to generate interpretation
            prompt = f"""
            Based on the following EEG analysis:
            State: {eeg_state.state}
            Confidence: {eeg_state.confidence:.2%}
            Features: {json.dumps(eeg_state.features)}
            
            Provide a professional medical interpretation:
            """
            
            interpretation = self.explainer(
                prompt,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7
            )[0]["generated_text"]
            
            # Clean up the generated text
            interpretation = interpretation.replace(prompt, "").strip()
            
            return interpretation
            
        except Exception as e:
            self.logger.error(f"Error generating interpretation: {str(e)}")
            raise
    
    def _generate_recommendations(self,
                                eeg_state: EEGState,
                                state_info: Dict) -> List[str]:
        """Generate recommendations based on the EEG state."""
        try:
            recommendations = []
            
            # Add state-specific recommendations
            if eeg_state.state.lower() == "calm":
                recommendations.extend([
                    "Maintain current relaxation techniques",
                    "Consider mindfulness exercises",
                    "Monitor stress levels regularly"
                ])
            elif eeg_state.state.lower() == "focused":
                recommendations.extend([
                    "Continue current focus-enhancing activities",
                    "Take regular breaks to prevent mental fatigue",
                    "Maintain good sleep hygiene"
                ])
            elif eeg_state.state.lower() == "drowsy":
                recommendations.extend([
                    "Consider taking a short rest",
                    "Ensure adequate sleep duration",
                    "Monitor caffeine intake"
                ])
            elif eeg_state.state.lower() == "stressed":
                recommendations.extend([
                    "Practice stress-reduction techniques",
                    "Consider professional consultation if stress persists",
                    "Implement regular relaxation exercises"
                ])
            
            # Add safety recommendations if needed
            for feature, value in eeg_state.features.items():
                if "amplitude" in feature.lower():
                    threshold = self.guidelines["safety_thresholds"].get(feature, 0)
                    if value > threshold:
                        recommendations.append(
                            f"Monitor {feature} levels and consult healthcare provider if elevated levels persist"
                        )
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            raise
    
    def _generate_temporal_analysis(self,
                                  eeg_state: EEGState,
                                  additional_context: Optional[Dict] = None) -> str:
        """Generate temporal analysis section."""
        try:
            analysis = "Temporal Analysis:\n"
            
            if additional_context and "previous_sessions" in additional_context:
                sessions = additional_context["previous_sessions"]
                analysis += f"Session History: This is session #{sessions + 1}\n"
                
                # Add session duration if available
                if "session_duration" in additional_context:
                    analysis += f"Current Session Duration: {additional_context['session_duration']}\n"
                
                # Add sleep information if available
                if "sleep_hours" in additional_context:
                    analysis += f"Reported Sleep: {additional_context['sleep_hours']}\n"
                    
                    # Add sleep-related recommendations
                    sleep_hours = float(additional_context['sleep_hours'].split()[0])
                    if sleep_hours < 7:
                        analysis += "Note: Sleep duration is below recommended 7-9 hours\n"
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error generating temporal analysis: {str(e)}")
            raise

    def _generate_safety_assessment(self,
                                  eeg_state: EEGState,
                                  additional_context: Optional[Dict] = None) -> Dict:
        """Generate safety assessment section."""
        try:
            assessment = {
                "alert_level": "mild",
                "concerns": [],
                "immediate_actions": []
            }
            
            # Determine occupation-specific thresholds
            occupation = additional_context.get("occupation", "default").lower().replace(" ", "_")
            
            # Check amplitude thresholds
            for feature, value in eeg_state.features.items():
                if "amplitude" in feature.lower():
                    threshold = self.guidelines["safety_thresholds"].get(
                        feature, {}).get(occupation, 
                        self.guidelines["safety_thresholds"].get(feature, {}).get("default", 0))
                    
                    if value > threshold:
                        assessment["concerns"].append(
                            f"Elevated {feature} levels detected"
                        )
                        
                        # Determine alert level
                        if value > threshold * 1.5:
                            assessment["alert_level"] = "severe"
                            assessment["immediate_actions"].append(
                                f"Urgent: {feature} levels significantly above safety threshold"
                            )
                        elif value > threshold * 1.2:
                            assessment["alert_level"] = "moderate"
                            assessment["immediate_actions"].append(
                                f"Warning: {feature} levels above safety threshold"
                            )
            
            # Add occupation-specific safety recommendations
            if occupation in ["software_engineer", "healthcare_worker"]:
                assessment["immediate_actions"].append(
                    "Consider taking a short break to reduce mental strain"
                )
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error generating safety assessment: {str(e)}")
            raise
    
    def format_explanation(self, explanation: Dict) -> str:
        """
        Format the explanation into a professional medical report.
        
        Parameters:
        -----------
        explanation : Dict
            The structured explanation dictionary
            
        Returns:
        --------
        str
            Formatted medical report
        """
        try:
            report = "EEG Analysis Report\n"
            report += "=" * 50 + "\n\n"
            
            # Add metadata
            report += f"Subject ID: {explanation['metadata']['subject_id']}\n"
            report += f"Session ID: {explanation['metadata']['session_id']}\n"
            report += f"Timestamp: {explanation['metadata']['timestamp']}\n"
            report += f"Confidence: {explanation['metadata']['confidence']:.2%}\n\n"
            
            # Add safety alert if present
            if explanation.get("safety_assessment", {}).get("alert_level") != "mild":
                report += "⚠️ ALERT: " + explanation["safety_assessment"]["alert_level"].upper() + "\n"
                report += "-" * 20 + "\n"
                for action in explanation["safety_assessment"]["immediate_actions"]:
                    report += f"• {action}\n"
                report += "\n"
            
            # Add main sections
            report += "1. Clinical Observation\n"
            report += "-" * 20 + "\n"
            report += explanation['clinical_observation'] + "\n\n"
            
            report += "2. Technical Analysis\n"
            report += "-" * 20 + "\n"
            report += explanation['technical_analysis'] + "\n\n"
            
            report += "3. Interpretation\n"
            report += "-" * 20 + "\n"
            report += explanation['interpretation'] + "\n\n"
            
            report += "4. Temporal Analysis\n"
            report += "-" * 20 + "\n"
            report += explanation['temporal_analysis'] + "\n\n"
            
            report += "5. Safety Assessment\n"
            report += "-" * 20 + "\n"
            for concern in explanation['safety_assessment']['concerns']:
                report += f"• {concern}\n"
            report += "\n"
            
            report += "6. Recommendations\n"
            report += "-" * 20 + "\n"
            for i, rec in enumerate(explanation['recommendations'], 1):
                report += f"{i}. {rec}\n"
            
            # Add additional context if available
            if "additional_context" in explanation:
                report += "\nAdditional Context\n"
                report += "-" * 20 + "\n"
                for key, value in explanation['additional_context'].items():
                    report += f"{key}: {value}\n"
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error formatting explanation: {str(e)}")
            raise 