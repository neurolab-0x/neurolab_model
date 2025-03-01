import os

import tensorflow

def save_trained_model(model, model_path):
    """
    Saves the trained EEG classification model to the specified path.

    Args:
        model (tensorflow.keras.Model): Trained model to be saved.
        model_path (str): Path where the model will be saved.

    Returns:
        None
    """
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        if not isinstance(model, tensorflow.keras.Model):
            raise TypeError("The provided model is not a valid tensorflow.keras.Model.")

        model.save(model_path)
        print(f"Model saved successfully at {model_path}")

    except Exception as e:
        print(f"An error occurred while saving the model: {e}")
