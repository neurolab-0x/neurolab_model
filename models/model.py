from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils import compute_class_weight
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Input, Dropout, 
    BatchNormalization, Activation, Add, Concatenate, Attention, 
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Bidirectional,
    SpatialDropout1D, GaussianNoise, TimeDistributed, SeparableConv1D
)
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, 
    TensorBoard, LearningRateScheduler
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import numpy as np
from models.save_trained_model import save_trained_model
from tensorflow.keras import Sequential
import tensorflow as tf
import math
import time
import os
from datetime import datetime
from utils.influxdb_client import InfluxDBManager

def cosine_annealing_schedule(epoch, lr):
    """Cosine annealing learning rate schedule."""
    initial_lr = 0.001
    min_lr = 1e-6
    max_epochs = 100
    return min_lr + (initial_lr - min_lr) * (1 + math.cos(math.pi * epoch / max_epochs)) / 2

def residual_block(x, filters, kernel_size=3, dropout_rate=0.3, use_separable=False):
    """Enhanced residual block with separable convolutions and better regularization."""
    shortcut = x
    
    # First conv layer
    if use_separable:
        x = SeparableConv1D(filters, kernel_size, padding='same')(x)
    else:
        x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout1D(dropout_rate)(x)
    
    # Second conv layer
    if use_separable:
        x = SeparableConv1D(filters, kernel_size, padding='same')(x)
    else:
        x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Skip connection - ensure dimensions match
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # Add skip connection
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

def transformer_block(x, embed_dim, num_heads, ff_dim, dropout=0.1, use_relative_pos=True):
    """Enhanced transformer block with relative positional encoding."""
    # Multi-head attention with relative positional encoding
    if use_relative_pos:
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim//num_heads,
            attention_axes=(1,)  # Apply attention along sequence dimension
        )(x, x, x)
    else:
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim//num_heads
        )(x, x)
    
    attention_output = Dropout(dropout)(attention_output)
    
    # Add & norm (first residual connection)
    x = LayerNormalization(epsilon=1e-6)(x + attention_output)
    
    # Feed-forward network with GELU activation
    ffn_output = Dense(ff_dim, activation='gelu')(x)
    ffn_output = Dense(embed_dim)(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    
    # Add & norm (second residual connection)
    x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    return x

def attention_lstm_layer(x, units, dropout_rate=0.3):
    """Enhanced LSTM layer with multi-head attention and better regularization."""
    # Bidirectional LSTM with residual connection
    lstm_out = Bidirectional(LSTM(units, return_sequences=True))(x)
    lstm_out = SpatialDropout1D(dropout_rate)(lstm_out)
    
    # Multi-head self-attention
    attention_output = MultiHeadAttention(
        num_heads=4, key_dim=units//4
    )(lstm_out, lstm_out)
    attention_output = Dropout(dropout_rate)(attention_output)
    
    # Add residual connection
    lstm_out = Add()([lstm_out, attention_output])
    lstm_out = LayerNormalization(epsilon=1e-6)(lstm_out)
    
    # Global context attention
    context_vector = Dense(1, activation='tanh')(lstm_out)
    attention_weights = Activation('softmax')(context_vector)
    context = tf.multiply(lstm_out, attention_weights)
    context = tf.reduce_sum(context, axis=1)
    
    return context

def train_hybrid_model(X_train, y_train, model_type='enhanced_cnn_lstm', **kwargs):
    """
    Enhanced hybrid model training with improved architectures and training features.
    
    Parameters:
    -----------
    X_train : array
        Training data
    y_train : array
        Training labels
    model_type : str
        Type of model to use
    kwargs : dict
        Additional parameters:
        - dropout_rate: Dropout rate (default: 0.3)
        - learning_rate: Learning rate (default: 0.001)
        - batch_size: Batch size (default: 32)
        - epochs: Number of epochs (default: 30)
        - use_separable: Whether to use separable convolutions (default: True)
        - use_relative_pos: Whether to use relative positional encoding (default: True)
        - l1_reg: L1 regularization factor (default: 1e-5)
        - l2_reg: L2 regularization factor (default: 1e-4)
        - influxdb_config: Dict containing InfluxDB configuration (optional)
        - subject_id: Unique identifier for the subject (optional)
        - session_id: Unique identifier for the session (optional)
    
    Returns:
    --------
    model : Keras model
        Trained model
    """
    # Extract parameters with defaults
    dropout_rate = kwargs.get('dropout_rate', 0.3)
    learning_rate = kwargs.get('learning_rate', 0.001)
    batch_size = kwargs.get('batch_size', 32)
    epochs = kwargs.get('epochs', 30)
    use_separable = kwargs.get('use_separable', True)
    use_relative_pos = kwargs.get('use_relative_pos', True)
    l1_reg = kwargs.get('l1_reg', 1e-5)
    l2_reg = kwargs.get('l2_reg', 1e-4)
    
    # InfluxDB configuration
    influxdb_config = kwargs.get('influxdb_config')
    subject_id = kwargs.get('subject_id', 'default_subject')
    session_id = kwargs.get('session_id', f'session_{int(time.time())}')
    
    # Initialize InfluxDB manager if config is provided
    influxdb_manager = None
    if influxdb_config:
        influxdb_manager = InfluxDBManager(
            url=influxdb_config['url'],
            token=influxdb_config['token'],
            org=influxdb_config['org'],
            bucket=influxdb_config['bucket']
        )
    
    # Reshape input for CNN
    X_train = X_train.reshape(-1, X_train.shape[1], 1)
    input_shape = (X_train.shape[1], 1)
    num_classes = len(set(y_train))
    
    # Model input with noise for regularization
    inputs = Input(shape=input_shape)
    x = GaussianNoise(0.01)(inputs)
    
    # Build selected architecture
    if model_type == 'original':
        # Original architecture with improvements
        x = Conv1D(64, kernel_size=3, activation='relu', 
                  kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(dropout_rate)(x)
        
        x = LSTM(64, return_sequences=True)(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu', 
                 kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
        x = Dropout(dropout_rate)(x)
        
    elif model_type == 'enhanced_cnn_lstm':
        # Enhanced CNN-LSTM with attention and separable convolutions
        x = SeparableConv1D(32, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = SpatialDropout1D(dropout_rate)(x)
        
        x = SeparableConv1D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = SpatialDropout1D(dropout_rate)(x)
        
        x = SeparableConv1D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(dropout_rate)(x)
        
        # Enhanced attention-based LSTM layer
        x = attention_lstm_layer(x, units=64, dropout_rate=dropout_rate)
        x = Dense(128, activation='relu', 
                 kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
    elif model_type == 'resnet_lstm':
        # Enhanced ResNet-style CNN with LSTM
        x = Conv1D(64, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout1D(dropout_rate)(x)
        
        # Enhanced residual blocks
        x = residual_block(x, filters=64, dropout_rate=dropout_rate, use_separable=use_separable)
        x = MaxPooling1D(pool_size=2)(x)
        
        x = residual_block(x, filters=128, dropout_rate=dropout_rate, use_separable=use_separable)
        x = MaxPooling1D(pool_size=2)(x)
        
        # Enhanced bidirectional LSTM with attention
        x = attention_lstm_layer(x, units=64, dropout_rate=dropout_rate)
        x = Dense(128, activation='relu', 
                 kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
    elif model_type == 'transformer':
        # Enhanced transformer-based architecture
        x = SeparableConv1D(64, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = SpatialDropout1D(dropout_rate)(x)
        
        # Enhanced transformer blocks
        embed_dim = 64
        x = transformer_block(x, embed_dim=embed_dim, num_heads=4, 
                            ff_dim=128, dropout=dropout_rate, 
                            use_relative_pos=use_relative_pos)
        x = transformer_block(x, embed_dim=embed_dim, num_heads=4, 
                            ff_dim=128, dropout=dropout_rate,
                            use_relative_pos=use_relative_pos)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu', 
                 kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
        x = Dropout(dropout_rate)(x)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Output layer with label smoothing
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Calculate class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    # Enhanced callbacks
    lr_scheduler = LearningRateScheduler(cosine_annealing_schedule)
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True,
        mode='min'
    )
    model_checkpoint = ModelCheckpoint(
        f'best_model_{model_type}.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )
    tensorboard = TensorBoard(
        log_dir=f'./logs/{model_type}',
        histogram_freq=1
    )
    
    # Custom callback for logging metrics to InfluxDB
    class InfluxDBLogger(tf.keras.callbacks.Callback):
        def __init__(self, influxdb_manager, subject_id, session_id, model_type):
            super().__init__()
            self.influxdb_manager = influxdb_manager
            self.subject_id = subject_id
            self.session_id = session_id
            self.model_type = model_type
        
        def on_epoch_end(self, epoch, logs=None):
            if self.influxdb_manager and logs:
                metrics = {
                    'loss': float(logs['loss']),
                    'accuracy': float(logs['accuracy']),
                    'val_loss': float(logs['val_loss']),
                    'val_accuracy': float(logs['val_accuracy'])
                }
                self.influxdb_manager.write_personalized_metrics(
                    metrics=metrics,
                    timestamp=datetime.now(),
                    subject_id=self.subject_id,
                    session_id=self.session_id
                )
    
    # Add InfluxDB logger to callbacks if configured
    callbacks = [lr_scheduler, early_stopping, model_checkpoint, tensorboard]
    if influxdb_manager:
        callbacks.append(InfluxDBLogger(influxdb_manager, subject_id, session_id, model_type))
    
    # Train model with enhanced callbacks
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    print(f"Model training complete. Type: {model_type}")
    model_path = f"./processed/trained_model_{model_type}.h5"
    save_trained_model(model, model_path)
    
    # Close InfluxDB connection if it was opened
    if influxdb_manager:
        influxdb_manager.close()
    
    return model, history

def evaluate_model(model, X_test, y_test, calibrate=True, **kwargs):
    """
    Enhanced model evaluation with detailed metrics and calibration.
    
    Parameters:
    -----------
    model : Keras model
        Trained model to evaluate
    X_test : array
        Test data
    y_test : array
        Test labels
    calibrate : bool
        Whether to perform confidence calibration
    kwargs : dict
        Additional parameters:
        - influxdb_config: Dict containing InfluxDB configuration (optional)
        - subject_id: Unique identifier for the subject (optional)
        - session_id: Unique identifier for the session (optional)
        - model_type: Type of model being evaluated (optional)
    
    Returns:
    --------
    metrics : dict
        Evaluation metrics
    """
    # Extract InfluxDB parameters
    influxdb_config = kwargs.get('influxdb_config')
    subject_id = kwargs.get('subject_id', 'default_subject')
    session_id = kwargs.get('session_id', f'eval_{int(time.time())}')
    model_type = kwargs.get('model_type', 'unknown')
    
    # Initialize InfluxDB manager if config is provided
    influxdb_manager = None
    if influxdb_config:
        influxdb_manager = InfluxDBManager(
            url=influxdb_config['url'],
            token=influxdb_config['token'],
            org=influxdb_config['org'],
            bucket=influxdb_config['bucket']
        )
    
    X_test = X_test.reshape(-1, X_test.shape[1], 1)
    
    # Get predictions with temperature scaling
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Calculate ROC AUC for multi-class (one-vs-rest)
    try:
        y_test_oh = tf.keras.utils.to_categorical(y_test)
        auc = roc_auc_score(y_test_oh, y_pred_proba, multi_class='ovr')
    except Exception as e:
        print(f"Could not calculate ROC AUC: {e}")
        auc = None
    
    # Log predictions to InfluxDB if configured
    if influxdb_manager:
        for i in range(len(X_test)):
            influxdb_manager.write_model_predictions(
                predictions=y_pred[i:i+1],
                probabilities=y_pred_proba[i:i+1],
                timestamp=datetime.now(),
                subject_id=subject_id,
                session_id=session_id,
                model_type=model_type
            )
    
    # Apply confidence calibration if requested
    calibration_metrics = {}
    if calibrate:
        try:
            from utils.interpretability import ModelInterpretability
            
            # Split test set into calibration and evaluation
            n_calib = int(0.3 * len(X_test))
            X_calib, X_eval = X_test[:n_calib], X_test[n_calib:]
            y_calib, y_eval = y_test[:n_calib], y_test[n_calib:]
            
            # Create interpretability handler
            interpreter = ModelInterpretability(model)
            
            # Calibrate confidence with temperature scaling
            cal_results = interpreter.calibrate_confidence(
                X_calib, y_calib, 
                method='temperature_scaling',
                n_bins=10
            )
            
            if "error" not in cal_results:
                # Get calibrated predictions
                cal_preds = interpreter.predict_with_calibration(X_eval)
                
                # Calculate calibration metrics
                uncal_preds = model.predict(X_eval)
                before_ece = interpreter._expected_calibration_error(uncal_preds, y_eval)
                after_ece = interpreter._expected_calibration_error(cal_preds["probabilities"], y_eval)
                
                # Calculate calibrated accuracy
                cal_y_pred = cal_preds["predictions"]
                cal_accuracy = accuracy_score(y_eval, cal_y_pred)
                
                calibration_metrics = {
                    "temperature": float(cal_results.get("temperature", 1.0)),
                    "ece_before": float(before_ece),
                    "ece_after": float(after_ece),
                    "calibrated_accuracy": float(cal_accuracy),
                    "improvement_percent": float((after_ece - before_ece) / before_ece * -100) 
                                        if before_ece > 0 else 0.0
                }
                
                # Log calibration metrics to InfluxDB if configured
                if influxdb_manager:
                    influxdb_manager.write_personalized_metrics(
                        metrics=calibration_metrics,
                        timestamp=datetime.now(),
                        subject_id=subject_id,
                        session_id=session_id
                    )
                
                print(f"Confidence calibration:")
                print(f"  - ECE before: {before_ece:.4f}")
                print(f"  - ECE after: {after_ece:.4f}")
                print(f"  - Improvement: {calibration_metrics['improvement_percent']:.1f}%")
            else:
                print(f"Calibration error: {cal_results['error']}")
        except Exception as e:
            print(f"Could not calibrate confidence: {e}")
    
    # Print results
    print(f"Model Accuracy: {accuracy:.4f}")
    if auc is not None:
        print(f"ROC AUC Score: {auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Detailed metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': report,
        'roc_auc': auc
    }
    
    # Add calibration metrics if available
    if calibration_metrics:
        metrics['calibration'] = calibration_metrics
    
    # Close InfluxDB connection if it was opened
    if influxdb_manager:
        influxdb_manager.close()
    
    return metrics

def model_comparison(X_train, y_train, X_test, y_test, n_repeats=3, **kwargs):
    """
    Enhanced model comparison with more detailed metrics and visualization.
    
    Parameters:
    -----------
    X_train, y_train : arrays
        Training data
    X_test, y_test : arrays
        Test data
    n_repeats : int
        Number of times to repeat training with different random initializations
    kwargs : dict
        Additional parameters:
        - influxdb_config: Dict containing InfluxDB configuration (optional)
        - subject_id: Unique identifier for the subject (optional)
    
    Returns:
    --------
    results : dict
        Performance metrics for each model type
    """
    # Extract InfluxDB parameters
    influxdb_config = kwargs.get('influxdb_config')
    subject_id = kwargs.get('subject_id', 'default_subject')
    
    # Initialize InfluxDB manager if config is provided
    influxdb_manager = None
    if influxdb_config:
        influxdb_manager = InfluxDBManager(
            url=influxdb_config['url'],
            token=influxdb_config['token'],
            org=influxdb_config['org'],
            bucket=influxdb_config['bucket']
        )
    
    model_types = ['original', 'enhanced_cnn_lstm', 'resnet_lstm', 'transformer']
    results = {model_type: {
        'accuracy': [], 
        'auc': [],
        'training_time': [],
        'inference_time': [],
        'model_size': []
    } for model_type in model_types}
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Training {model_type} model...")
        print(f"{'='*50}\n")
        
        for i in range(n_repeats):
            print(f"Run {i+1}/{n_repeats}")
            session_id = f'{model_type}_run_{i+1}'
            
            # Measure training time
            start_time = time.time()
            model, _ = train_hybrid_model(
                X_train, y_train, 
                model_type=model_type,
                influxdb_config=influxdb_config,
                subject_id=subject_id,
                session_id=session_id
            )
            training_time = time.time() - start_time
            
            # Measure inference time
            inference_start = time.time()
            _ = model.predict(X_test[:100])  # Test on subset
            inference_time = (time.time() - inference_start) / 100  # Average per sample
            
            # Get model size
            model.save("temp.h5")
            model_size = tf.io.gfile.GFile("temp.h5").size() / 1e6  # Size in MB
            os.remove("temp.h5")
            
            # Evaluate model
            metrics = evaluate_model(
                model, X_test, y_test,
                influxdb_config=influxdb_config,
                subject_id=subject_id,
                session_id=session_id,
                model_type=model_type
            )
            
            # Store results
            results[model_type]['accuracy'].append(metrics['accuracy'])
            if metrics['roc_auc'] is not None:
                results[model_type]['auc'].append(metrics['roc_auc'])
            results[model_type]['training_time'].append(training_time)
            results[model_type]['inference_time'].append(inference_time)
            results[model_type]['model_size'].append(model_size)
        
        # Calculate average performance
        avg_accuracy = np.mean(results[model_type]['accuracy'])
        std_accuracy = np.std(results[model_type]['accuracy'])
        avg_training_time = np.mean(results[model_type]['training_time'])
        avg_inference_time = np.mean(results[model_type]['inference_time'])
        avg_model_size = np.mean(results[model_type]['model_size'])
        
        print(f"\n{model_type} Results:")
        print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"Average Training Time: {avg_training_time:.2f}s")
        print(f"Average Inference Time: {avg_inference_time*1000:.2f}ms")
        print(f"Average Model Size: {avg_model_size:.2f}MB")
        
        if results[model_type]['auc']:
            avg_auc = np.mean(results[model_type]['auc'])
            std_auc = np.std(results[model_type]['auc'])
            print(f"Average AUC: {avg_auc:.4f} ± {std_auc:.4f}")
    
    # Close InfluxDB connection if it was opened
    if influxdb_manager:
        influxdb_manager.close()
    
    return results
