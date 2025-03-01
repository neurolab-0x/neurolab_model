from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.utils import compute_class_weight
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Input, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau # type: ignore
import numpy as np
from models.save_trained_model import save_trained_model
from tensorflow.keras import Sequential # type: ignore

def train_hybrid_model(X_train, y_train):
    """Trains a hybrid CNN-LSTM model for EEG classification."""
    X_train = X_train.reshape(-1, X_train.shape[1], 1)
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        LSTM(64, return_sequences=True),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(set(y_train)), activation='softmax')
    ])
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, class_weight=class_weight_dict, callbacks=[lr_scheduler])
    print(f"Evaluated model : {model}")
    model_path = "./processed/trained_model.h5"
    save_trained_model(model, model_path)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model on test data."""
    X_test = X_test.reshape(-1, X_test.shape[1], 1)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)
    return accuracy, report
