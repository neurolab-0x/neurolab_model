import numpy as np
from preprocessing.preprocess import preprocess_data
from models.load_trained_model import load_trained_model

def process_realtime_data(data):
    """
    Processes incoming real-time EEG data and returns model predictions.
    :param data: Raw EEG data in numpy array or list format.
    :return: Predicted mental state and confidence score.
    """
    try:
        # Convert data to numpy array if not already
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Ensure data shape consistency
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Preprocess the real-time data
        X_real = preprocess_data(data)
        
        # Load trained model
        
        model_path="./data/processed/trained_model.h5"
        model = load_trained_model(model_path)
        
        # Reshape data for model input
        X_real = X_real.reshape(-1, X_real.shape[1], 1)
        
        # Get model predictions
        predictions = model.predict(X_real)
        predicted_state = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        
        return {"predicted_state": int(predicted_state), "confidence": float(confidence)}
    
    except Exception as e:
        return {"error": str(e)}
