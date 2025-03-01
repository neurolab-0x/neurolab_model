from config.settings import THRESHOLDS

def generate_recommendations(state_durations, total_duration, confidence):
    """Generate clinical recommendations based on state analysis"""
    recommendations = []
    
    stress_ratio = state_durations.get(2, 0) / total_duration
    if stress_ratio > THRESHOLDS['stress_threshold'] and confidence > THRESHOLDS['confidence_threshold']:
        base_rec = "🧠 Prolonged cognitive load detected: Try 4-7-8 breathing technique"
        if stress_ratio > THRESHOLDS['severe_stress_threshold']:
            base_rec += " with binaural beats audio support"
        recommendations.append(base_rec)
    
    if state_durations.get(0, 0) / total_duration > THRESHOLDS['relaxation_threshold']:
        recommendations.append("🕊️ Maintain current activities - optimal relaxation state detected")
    
    if not recommendations:
        recommendations.append("🌱 General mindfulness exercise recommended: Focus on diaphragmatic breathing")
    
    return recommendations[:THRESHOLDS['max_recommendations']]
