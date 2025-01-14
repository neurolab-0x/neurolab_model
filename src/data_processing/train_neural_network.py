import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

# Function to load and clean a single dataset
def load_and_clean_data(file_path):
    dataframe = pd.read_csv(file_path)
    dataframe = dataframe.dropna()  # Remove rows with missing values
    return dataframe

# Function to remove outliers
def remove_outliers(dataframe, z_thresh=3):
    z_scores = np.abs((dataframe - dataframe.mean()) / dataframe.std())
    filtered = dataframe[(z_scores < z_thresh).all(axis=1)]
    print(f"Outliers removed: {len(dataframe) - len(filtered)} rows")
    return filtered

# Function to scale features
def scale_features(dataframe):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataframe)
    return pd.DataFrame(scaled_data, columns=dataframe.columns)

# Function to handle class imbalance using SMOTE
def handle_class_imbalance(x_train, y_train):
    smote = SMOTE()
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)
    print("Class distribution after SMOTE:")
    print(pd.DataFrame(y_train_balanced, columns=['mental_state']).value_counts())
    return x_train_balanced, y_train_balanced

# Function to split data into training and testing sets
# Function to split data into training and testing sets
def split_data(dataframe):
    x = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]

    # Remove rows where labels are NaN
    valid_indices = ~y.isna()
    x = x[valid_indices]
    y = y[valid_indices]

    return train_test_split(x, y, test_size=0.2, random_state=42)

# Function to handle class imbalance using SMOTE
def handle_class_imbalance(x_train, y_train):
    print("Checking for NaN in y_train before SMOTE:")
    print(y_train.isna().sum())
    
    smote = SMOTE()
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)
    
    print("Class distribution after SMOTE:")
    print(pd.DataFrame(y_train_balanced, columns=['mental_state']).value_counts())
    
    return x_train_balanced, y_train_balanced

# Function to extract features
def extract_features(dataframe):
    extracted_features = pd.DataFrame()

    for column in dataframe.columns[:-1]:
        channel_data = dataframe[column]

        freqs, psd = welch(channel_data, fs=128)
        extracted_features[f'{column}_delta_power'] = [np.sum(psd[(freqs >= 0.5) & (freqs < 4)])]
        extracted_features[f'{column}_theta_power'] = [np.sum(psd[(freqs >= 4) & (freqs < 8)])]
        extracted_features[f'{column}_alpha_power'] = [np.sum(psd[(freqs >= 8) & (freqs < 13)])]
        extracted_features[f'{column}_beta_power'] = [np.sum(psd[(freqs >= 13) & (freqs < 30)])]
        extracted_features[f'{column}_gamma_power'] = [np.sum(psd[(freqs >= 30) & (freqs < 50)])]

        extracted_features[f'{column}_shannon_entropy'] = [-np.sum(psd * np.log(psd + 1e-9))]
        extracted_features[f'{column}_mean'] = [channel_data.mean()]
        extracted_features[f'{column}_std'] = [channel_data.std()]
        extracted_features[f'{column}_skew'] = [skew(channel_data)]
        extracted_features[f'{column}_kurtosis'] = [kurtosis(channel_data)]
        extracted_features[f'{column}_max'] = [channel_data.max()]
        extracted_features[f'{column}_min'] = [channel_data.min()]
        extracted_features[f'{column}_range'] = [channel_data.max() - channel_data.min()]
        extracted_features[f'{column}_fractal_dimension'] = [np.log(len(channel_data)) / np.log(len(np.unique(channel_data)))]

        for other_column in dataframe.columns[:-1]:
            if column != other_column:
                extracted_features[f'{column}_coherence_with_{other_column}'] = [np.corrcoef(channel_data, dataframe[other_column])[0, 1]]

    return extracted_features

# Function to evaluate a model
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred_classes = y_pred.argmax(axis=1) if hasattr(y_pred, 'argmax') else y_pred

    print("Classification Report:")
    print(classification_report(y_test, y_pred_classes))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_classes))

    accuracy = accuracy_score(y_test, y_pred_classes)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Neural network construction
def build_neural_network(input_dim, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train a neural network
def train_neural_network(x_train, y_train, x_test, y_test, num_classes, epochs=20, batch_size=32):
    y_train_encoded = to_categorical(y_train, num_classes)
    y_test_encoded = to_categorical(y_test, num_classes)

    model = build_neural_network(input_dim=x_train.shape[1], num_classes=num_classes)

    model.fit(x_train, y_train_encoded,
              validation_data=(x_test, y_test_encoded),
              epochs=epochs,
              batch_size=batch_size,
              verbose=1)

    return evaluate_model(model, x_test, y_test)

# Function to load multiple datasets
def load_multiple_datasets(directory):
    combined_dataframe = pd.DataFrame()

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            print(f"Loading dataset: {filename}")
            dataframe = load_and_clean_data(file_path)
            dataframe = remove_outliers(dataframe)

            if not dataframe.empty:
                combined_dataframe = pd.concat([combined_dataframe, dataframe], ignore_index=True)
            else:
                print(f"Dataset {filename} is empty after outlier removal. Skipping...")

    print("All datasets loaded and combined.")
    print(f"Combined dataset shape: {combined_dataframe.shape}")
    return combined_dataframe

# Main execution function
def main():
    dataset_directory = '../../data/raw/'
    combined_dataframe = load_multiple_datasets(dataset_directory)

    x_train, x_test, y_train, y_test = split_data(combined_dataframe)

    print("Data preprocessing and splitting completed.")
    print(f"Training features shape: {x_train.shape}")
    print(f"Testing features shape: {x_test.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Testing labels shape: {y_test.shape}")

    x_train_balanced, y_train_balanced = handle_class_imbalance(x_train, y_train)

    print("Training neural network...")
    train_neural_network(x_train_balanced, y_train_balanced, x_test, y_test, num_classes=3)

if __name__ == "__main__":
    main()
