import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

def load_and_clean_data(file_path):
    """
    Load and preprocess the EEG data.

    Args:
        file_path (str): Path to the raw EEG data file.

    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame.
    """
    raw_dataframe = pd.read_csv(file_path)
    print(f"Dataset loaded: {file_path}, Shape: {raw_dataframe.shape}")
    dataframe = raw_dataframe.drop(columns=['Unnamed: 32'], errors='ignore')

    dataframe = dataframe.fillna(dataframe.median())

    return dataframe

def scale_features(dataframe):
    """
    Scale features using StandardScaler.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Scaled DataFrame.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataframe)
    scaled_dataframe = pd.DataFrame(scaled_data, columns=dataframe.columns)

    return scaled_dataframe

def remove_outliers(dataframe, z_threshold=3):
    """
    Remove outliers based on z-scores.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        z_threshold (float): Z-score threshold to identify outliers.

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    z_scores = np.abs((dataframe - dataframe.mean()) / dataframe.std())
    filtered_dataframe = dataframe[(z_scores < z_threshold).all(axis=1)]
    print(f"Outliers removed: {len(dataframe) - len(filtered_dataframe)} rows")
    return filtered_dataframe

def add_mental_state_labels(dataframe):
    """
    Add mental state labels to the DataFrame based on mean EEG values.

    Args:
        dataframe (pd.DataFrame): Input EEG data.

    Returns:
        pd.DataFrame: DataFrame with a new column 'mental_state'.
    """
    eeg_columns = dataframe.columns[:-1]
    mean_values = dataframe[eeg_columns].mean(axis=1)

    def determine_mental_state(mean_value):
        if mean_value < -0.1:
            return 'Calm'
        elif mean_value > 0.1:
            return 'Stressed'
        else:
            return 'Neutral'

    dataframe['mental_state'] = mean_values.apply(determine_mental_state)
    return dataframe

def load_multiple_datasets(dataset_directory):
    """
    Load and preprocess multiple datasets from a directory, merging them into a single DataFrame.

    Args:
        dataset_directory (str): Path to the directory containing dataset files.

    Returns:
        pd.DataFrame: Combined and preprocessed DataFrame.
    """
    combined_dataframe = pd.DataFrame()
    reference_columns = None
    missing_columns_summary = {}
    extra_columns_summary = {}

    for file_name in os.listdir(dataset_directory):
        file_path = os.path.join(dataset_directory, file_name)
        if file_name.endswith('.csv'):
            print(f"Loading dataset: {file_name}")
            dataframe = load_and_clean_data(file_path)

            # Check and align column structure
            if reference_columns is None:
                reference_columns = dataframe.columns
            else:
                missing_cols = set(reference_columns) - set(dataframe.columns)
                extra_cols = set(dataframe.columns) - set(reference_columns)

                if missing_cols:
                    missing_columns_summary[file_name] = missing_cols
                    for col in missing_cols:
                        dataframe[col] = np.nan

                if extra_cols:
                    extra_columns_summary[file_name] = extra_cols
                    dataframe = dataframe.drop(columns=extra_cols, errors='ignore')

                dataframe = dataframe[reference_columns]

            dataframe = remove_outliers(dataframe)

            if dataframe.empty:
                print(f"Dataset {file_name} is empty after outlier removal. Skipping...")
                continue

            dataframe = scale_features(dataframe)

            dataframe = add_mental_state_labels(dataframe)

            dataframe['dataset_source'] = file_name

            try:
                combined_dataframe = pd.concat([combined_dataframe, dataframe], ignore_index=True)
            except MemoryError:
                print("MemoryError: Unable to concatenate the dataset. Skipping...")

    print("All datasets loaded and combined.")
    print(f"Combined dataset shape: {combined_dataframe.shape}")

    if missing_columns_summary:
        print("Summary of Missing Columns:")
        for file, cols in missing_columns_summary.items():
            print(f"  {file}: {cols}")

    if extra_columns_summary:
        print("Summary of Extra Columns:")
        for file, cols in extra_columns_summary.items():
            print(f"  {file}: {cols}")

    return combined_dataframe

def split_data(dataframe):
    """
    Split the data into training and testing sets.

    Args:
        dataframe (pd.DataFrame): Input data with features and labels.

    Returns:
        tuple: Training and testing features (X_train, X_test) and labels (y_train, y_test).
    """
    x = dataframe.drop(columns=['mental_state', 'dataset_source'], errors='ignore')
    y = dataframe['mental_state']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test

def train_random_forest_with_grid_search(x_train, y_train):
    """
    Train a Random Forest Classifier using grid search for hyperparameter tuning.

    Args:
        x_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        RandomForestClassifier: Trained model with the best parameters.
        dict: Best parameters from grid search.
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                               param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print("Best Parameters from Grid Search:", best_params)

    return best_model, best_params

def cross_validate_model(model, x, y, cv=5):
    """
    Perform k-fold cross-validation on the given model.

    Args:
        model (RandomForestClassifier): Trained model.
        x (pd.DataFrame): Features.
        y (pd.Series): Labels.
        cv (int): Number of cross-validation folds.

    Returns:
        None
    """
    print("Performing k-fold cross-validation...")
    scores = cross_val_score(model, x, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Cross-Validation Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.2f}")
    print(f"Standard Deviation: {scores.std():.2f}")

def evaluate_model(model, x_test, y_test):
    """
    Evaluate the trained model on the test data.

    Args:
        model (RandomForestClassifier): Trained model.
        x_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing labels.

    Returns:
        None
    """
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")

def plot_feature_importance(model, feature_names):
    """
    Plot the feature importance of the trained Random Forest model.

    Args:
        model (RandomForestClassifier): Trained model.
        feature_names (list): List of feature names.

    Returns:
        None
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances - EEG Channels")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel("EEG Channels")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.show()

def save_model(model, file_path):
    """
    Save the trained model to a file.

    Args:
        model (RandomForestClassifier): Trained model.
        file_path (str): Path to save the model.

    Returns:
        None
    """
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

def load_model(file_path):
    """
    Load a model from a file.

    Args:
        file_path (str): Path to the saved model file.

    Returns:
        RandomForestClassifier: Loaded model.
    """
    model = joblib.load(file_path)
    print(f"Model loaded from {file_path}")
    return model

def main():
    dataset_directory = '../../data/raw/'
    model_save_path = './random_forest_model.pkl'

    combined_dataframe = load_multiple_datasets(dataset_directory)

    x_train, x_test, y_train, y_test = split_data(combined_dataframe)

    print("Data preprocessing and splitting completed.")
    print(f"Training features shape: {x_train.shape}")
    print(f"Testing features shape: {x_test.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Testing labels shape: {y_test.shape}")

    model, best_params = train_random_forest_with_grid_search(x_train, y_train)
    print("Random Forest model trained with optimized hyperparameters.")

    evaluate_model(model, x_test, y_test)

    cross_validate_model(model, pd.concat([x_train, x_test]), pd.concat([y_train, y_test]))

    feature_names = x_train.columns.tolist()
    plot_feature_importance(model, feature_names)

    save_model(model, model_save_path)

    loaded_model = load_model(model_save_path)
    evaluate_model(loaded_model, x_test, y_test)

if __name__ == "__main__":
    main()
