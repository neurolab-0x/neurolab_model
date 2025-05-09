from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils import compute_class_weight
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Input, Dropout, 
    BatchNormalization, Activation, Add, Concatenate, Attention, 
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Bidirectional
) # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import numpy as np
from models.save_trained_model import save_trained_model
from tensorflow.keras import Sequential # type: ignore
import tensorflow as tf

def residual_block(x, filters, kernel_size=3):
    """Create a residual block with skip connection."""
    shortcut = x
    
    # First conv layer
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Second conv layer
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

def transformer_block(x, embed_dim, num_heads, ff_dim, dropout=0.1):
    """Create a transformer block with multi-head attention."""
    # Multi-head attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim//num_heads
    )(x, x)
    attention_output = Dropout(dropout)(attention_output)
    
    # Add & norm (first residual connection)
    x = LayerNormalization(epsilon=1e-6)(x + attention_output)
    
    # Feed-forward network
    ffn_output = Dense(ff_dim, activation='relu')(x)
    ffn_output = Dense(embed_dim)(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    
    # Add & norm (second residual connection)
    x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    return x

def attention_lstm_layer(x, units):
    """Create an LSTM layer with attention mechanism."""
    lstm_out = Bidirectional(LSTM(units, return_sequences=True))(x)
    
    # Self-attention mechanism
    attention = Dense(1, activation='tanh')(lstm_out)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = tf.expand_dims(attention, -1)
    
    # Apply attention weights to LSTM output
    context = tf.multiply(lstm_out, attention)
    context = tf.reduce_sum(context, axis=1)
    
    return context

def train_hybrid_model(X_train, y_train, model_type='enhanced_cnn_lstm', **kwargs):
    """
    Trains various hybrid model architectures for EEG classification.
    
    Parameters:
    -----------
    X_train : array
        Training data
    y_train : array
        Training labels
    model_type : str
        Type of model to use:
        - 'enhanced_cnn_lstm': Enhanced CNN-LSTM with attention
        - 'resnet_lstm': CNN-LSTM with residual connections
        - 'transformer': Transformer-based architecture
        - 'original': Original CNN-LSTM architecture (for backward compatibility)
    kwargs : dict
        Additional parameters:
        - dropout_rate: Dropout rate (default: 0.3)
        - learning_rate: Learning rate (default: 0.001)
        - batch_size: Batch size (default: 32)
        - epochs: Number of epochs (default: 30)
    
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
    
    # Reshape input for CNN
    X_train = X_train.reshape(-1, X_train.shape[1], 1)
    input_shape = (X_train.shape[1], 1)
    num_classes = len(set(y_train))
    
    # Model input
    inputs = Input(shape=input_shape)
    
    # Build selected architecture
    if model_type == 'original':
        # Original architecture (for backward compatibility)
        x = Conv1D(64, kernel_size=3, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = BatchNormalization()(x)
        x = LSTM(64, return_sequences=True)(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        
    elif model_type == 'enhanced_cnn_lstm':
        # Enhanced CNN-LSTM with attention
        # CNN layers with increasing filters
        x = Conv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        
        # Attention-based LSTM layer
        x = attention_lstm_layer(x, units=64)
        x = Dropout(dropout_rate)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
    elif model_type == 'resnet_lstm':
        # ResNet-style CNN with LSTM
        x = Conv1D(64, kernel_size=3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Residual blocks
        x = residual_block(x, filters=64)
        x = MaxPooling1D(pool_size=2)(x)
        
        x = residual_block(x, filters=128)
        x = MaxPooling1D(pool_size=2)(x)
        
        # Bidirectional LSTM
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
    elif model_type == 'transformer':
        # Transformer-based architecture
        # Initial convolution to reduce sequence length
        x = Conv1D(64, kernel_size=3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        # Position embeddings would be added here in a full transformer implementation
        
        # Transformer blocks
        embed_dim = 64  # Must match the last dimension from Conv1D
        x = transformer_block(x, embed_dim=embed_dim, num_heads=4, ff_dim=128, dropout=dropout_rate)
        x = transformer_block(x, embed_dim=embed_dim, num_heads=4, ff_dim=128, dropout=dropout_rate)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Calculate class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    # Callbacks
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer, 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=[lr_scheduler, early_stopping],
        verbose=1
    )
    
    # Save model
    print(f"Model training complete. Type: {model_type}")
    model_path = f"./processed/trained_model_{model_type}.h5"
    save_trained_model(model, model_path)
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model on test data with detailed metrics."""
    X_test = X_test.reshape(-1, X_test.shape[1], 1)
    
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Calculate ROC AUC for multi-class (one-vs-rest)
    try:
        # One-hot encode true labels
        y_test_oh = tf.keras.utils.to_categorical(y_test)
        auc = roc_auc_score(y_test_oh, y_pred_proba, multi_class='ovr')
    except Exception as e:
        print(f"Could not calculate ROC AUC: {e}")
        auc = None
    
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
    
    return metrics

def model_comparison(X_train, y_train, X_test, y_test, n_repeats=3):
    """
    Compare performance of different model architectures.
    
    Parameters:
    -----------
    X_train, y_train : arrays
        Training data
    X_test, y_test : arrays
        Test data
    n_repeats : int
        Number of times to repeat training with different random initializations
    
    Returns:
    --------
    results : dict
        Performance metrics for each model type
    """
    model_types = ['original', 'enhanced_cnn_lstm', 'resnet_lstm', 'transformer']
    results = {model_type: {'accuracy': [], 'auc': []} for model_type in model_types}
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Training {model_type} model...")
        print(f"{'='*50}\n")
        
        for i in range(n_repeats):
            print(f"Run {i+1}/{n_repeats}")
            model, _ = train_hybrid_model(X_train, y_train, model_type=model_type)
            metrics = evaluate_model(model, X_test, y_test)
            
            results[model_type]['accuracy'].append(metrics['accuracy'])
            if metrics['roc_auc'] is not None:
                results[model_type]['auc'].append(metrics['roc_auc'])
        
        # Calculate average performance
        avg_accuracy = np.mean(results[model_type]['accuracy'])
        std_accuracy = np.std(results[model_type]['accuracy'])
        
        print(f"\n{model_type} - Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        
        if results[model_type]['auc']:
            avg_auc = np.mean(results[model_type]['auc'])
            std_auc = np.std(results[model_type]['auc'])
            print(f"{model_type} - Average AUC: {avg_auc:.4f} ± {std_auc:.4f}")
    
    return results
