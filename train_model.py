import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import os
import logging
from core.data.handler import DataHandler
from core.ml.processing import load_data, label_eeg_states, extract_features, preprocess_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_model(input_shape=(5, 1)):
    """Create a simple CNN model for EEG state classification"""
    model = models.Sequential([
        layers.Conv1D(32, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(3, activation='softmax')  # 3 states: relaxation, attention, stress
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_training_data(data_path: str):
    """Prepare data for training"""
    try:
        logger.info("Loading data...")
        # Load data using pandas
        df = pd.read_csv(data_path)
        
        # Extract features and labels
        X = df[['alpha', 'beta', 'theta', 'delta', 'gamma']].values
        y = df['state'].values
        
        # Normalize features
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / (X_std + 1e-10)
        
        # Split into train/test sets (80/20)
        split_idx = int(0.8 * len(X_normalized))
        X_train = X_normalized[:split_idx]
        X_test = X_normalized[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        # Reshape for CNN
        X_train = X_train.reshape(-1, X_train.shape[1], 1)
        X_test = X_test.reshape(-1, X_test.shape[1], 1)
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

def train_model(data_path: str, model_save_path: str):
    """Train the model using the provided data"""
    try:
        logger.info("Preparing training data...")
        X_train, X_test, y_train, y_test = prepare_training_data(data_path)
        
        # Create and train model
        logger.info("Creating model...")
        model = create_model(input_shape=(X_train.shape[1], 1))
        
        logger.info("Training model...")
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test accuracy: {test_acc:.4f}")
        
        # Save model
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        
        return model, history
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def main():
    """Main training script"""
    try:
        # Use sample data for training
        data_path = "test_data/sample_eeg.csv"
        model_save_path = "processed/trained_model.h5"
        
        logger.info(f"Starting model training with data from {data_path}")
        model, history = train_model(data_path, model_save_path)
        
        # Print training summary
        logger.info("Training completed successfully")
        logger.info(f"Final model saved to {model_save_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 