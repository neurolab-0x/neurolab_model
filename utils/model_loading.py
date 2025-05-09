# utils/model_loading.py
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model

class TemperatureScaling(Layer):
    """Custom layer for model calibration"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.temp = self.add_weight(
            name='temperature',
            shape=(1,),
            initializer='ones',
            trainable=False
        )

    def call(self, inputs):
        return inputs / self.temp

class AttentionLayer(Layer):
    """Custom attention layer"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform", 
            trainable=True
        )
        self.b = self.add_weight(
            name="attention_bias", 
            shape=(1,), 
            initializer="zeros", 
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        e = tf.tanh(tf.matmul(inputs, self.W) + self.b)  # (batch_size, time_steps, 1)
        a = tf.nn.softmax(e, axis=1)  # Attention weights
        context = inputs * a  # Apply attention weights
        context = tf.reduce_sum(context, axis=1)  # Sum over time steps
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def load_calibrated_model(path, model_type='original'):
    """
    Load model with custom layers
    
    Parameters:
    -----------
    path : str
        Path to the model file
    model_type : str
        Type of model architecture (used to determine custom objects)
    
    Returns:
    --------
    model : tf.keras.Model
        Loaded model
    """
    # Define all custom objects that might be in the model
    custom_objects = {
        'TemperatureScaling': TemperatureScaling,
        'AttentionLayer': AttentionLayer,
    }
    
    try:
        # Try loading with custom objects
        model = load_model(path, custom_objects=custom_objects)
        print(f"Model loaded successfully from {path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        # If loading fails, try without custom objects
        try:
            model = load_model(path)
            print(f"Model loaded successfully without custom objects from {path}")
            return model
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            raise

def get_available_models(directory="./processed/"):
    """Find all available trained models in the directory"""
    import glob
    import os
    
    # Find all h5 files
    model_files = glob.glob(os.path.join(directory, "trained_model*.h5"))
    
    # Extract model types
    models = []
    for file_path in model_files:
        filename = os.path.basename(file_path)
        if "_" in filename:
            # Extract model type from filename
            model_type = filename.split("_")[-1].replace(".h5", "")
            models.append({
                "path": file_path,
                "type": model_type,
                "filename": filename
            })
        else:
            # Original model without type
            models.append({
                "path": file_path,
                "type": "original",
                "filename": filename
            })
    
    return models
