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

def load_calibrated_model(path):
    """Load model with custom layer"""
    return load_model(
        path,
        custom_objects={'TemperatureScaling': TemperatureScaling}
    )
