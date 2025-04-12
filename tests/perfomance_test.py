import time
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load your trained model
model = tf.keras.models.load_model("processed/trained_model.h5")

# Dummy EEG test dataset (Replace with actual test data)
def generate_dummy_data(num_samples=100, sequence_length=10, num_classes=3):
    """Generate dummy EEG data with correct shape for Conv1D layer.
    
    Args:
        num_samples: Number of samples to generate
        sequence_length: Length of each EEG sequence (must be 10 to match model)
    
    Returns:
        X: Data with shape (num_samples, sequence_length, 1)
        y: Labels with shape (num_samples,)
    """
    X = np.random.rand(num_samples, sequence_length, 1).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    return X, y

X_test, y_test = generate_dummy_data()
print(f"Generated data shape: {X_test.shape}")
print(f"Model input shape: {model.input_shape}")

def evaluate_model(model, X_test, y_test, batch_size=16):
    inference_times = []
    y_preds = []
    
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size]
        start_time = time.time()
        outputs = model.predict(batch)
        inference_times.append(time.time() - start_time)
        
        preds = np.argmax(outputs, axis=1)
        y_preds.extend(preds)
    
    # Compute performance metrics with 'weighted' average for multiclass
    accuracy = accuracy_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds, average='weighted')
    recall = recall_score(y_test, y_preds, average='weighted')
    f1 = f1_score(y_test, y_preds, average='weighted')
    avg_inference_time = np.mean(inference_times)
    
    # Model size
    model.save("temp.h5")
    model_size = round((tf.io.gfile.GFile("temp.h5").size() / 1e6), 2)  # Size in MB
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Avg Inference Time (s)": avg_inference_time,
        "Model Size (MB)": model_size
    }

results = evaluate_model(model, X_test, y_test)
print("Benchmark Results:")
for key, value in results.items():
    print(f"{key}: {value}")
