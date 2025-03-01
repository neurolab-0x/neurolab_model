import tensorflow as tf
import os

def load_trained_model(model_path):
    """
    Loads the trained EEG classification model from the specified path.

    Args:
        model_path (str): Path to the trained model file.

    Returns:
        tensorflow.keras.Model: Loaded model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model loaded successfully.")
    return model
