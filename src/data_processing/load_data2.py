import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
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
from xgboost import XGBClassifier

# Function to load and clean data
def load_and_clean_data(file_path):
    dataframe = pd.read_csv(file_path)
    dataframe.dropna(inplace=True)  # Drop missing values
    return dataframe

# Function to remove outliers
def remove_outliers(dataframe):
    initial_shape = dataframe.shape
    for column in dataframe.select_dtypes(include=[np.number]).columns:
        Q1 = dataframe[column].quantile(0.25)
        Q3 = dataframe[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        dataframe = dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]
    print(f"Outliers removed: {initial_shape[0] - dataframe.shape[0]} rows")
    return dataframe

# Function to scale features
def scale_features(dataframe):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataframe)
    return pd.DataFrame(scaled_data, columns=dataframe.columns)

# Function to add mental state labels
def add_mental_state_labels(dataframe):
    dataframe['mental_state'] = np.random.choice(['Calm', 'Neutral', 'Stressed'], size=len(dataframe))
    return dataframe

# Function to extract advanced features
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

# Function to split data
def split_data(dataframe):
    x = dataframe.drop(columns=['mental_state'], errors='ignore')
    y = dataframe.get('mental_state')
    if y is None or x.empty:
        raise ValueError("Dataset is empty or missing the 'mental_state' column.")
    return train_test_split(x, y, test_size=0.2, random_state=42)

# Function to handle class imbalance
def handle_class_imbalance(x_train, y_train):
    smote = SMOTE(random_state=42)
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)
    return x_train_balanced, y_train_balanced

# Function to train Random Forest with GridSearchCV
def train_random_forest_with_grid_search(x_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', verbose=1)
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

# Function to train XGBoost
def train_xgboost(x_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(x_train, y_train)
    return model

# Function to train Stacking Classifier
def train_stacking(x_train, y_train):
    base_learners = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
    ]
    meta_learner = LogisticRegression()
    model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5)
    model.fit(x_train, y_train)
    return model

# Function to train Voting Classifier
def train_voting(x_train, y_train):
    base_learners = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
    ]
    model = VotingClassifier(estimators=base_learners, voting='soft')
    model.fit(x_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Function for cross-validation
def cross_validate_model(model, x, y):
    scores = cross_val_score(model, x, y, cv=5, scoring='accuracy')
    print("Cross-Validation Scores:", scores)
    print("Mean Accuracy:", scores.mean())
    print("Standard Deviation:", scores.std())

# Function to plot feature importance
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

# Function to save model
def save_model(model, file_path):
    joblib.dump(model, file_path)

# Function to load model
def load_model(file_path):
    return joblib.load(file_path)

# Updated load_multiple_datasets function
def load_multiple_datasets(dataset_directory):
    combined_dataframe = pd.DataFrame()
    reference_columns = None
    missing_columns_summary = {}
    extra_columns_summary = {}

    for file_name in os.listdir(dataset_directory):
        file_path = os.path.join(dataset_directory, file_name)
        if file_name.endswith('.csv'):
            print(f"Loading dataset: {file_name}")
            dataframe = load_and_clean_data(file_path)

            if dataframe.empty:
                print(f"Dataset {file_name} is empty after loading and cleaning. Skipping...")
                continue

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
            features = extract_features(dataframe)
            features['dataset_source'] = file_name

            try:
                combined_dataframe = pd.concat([combined_dataframe, features], ignore_index=True)
            except MemoryError:
                print("MemoryError: Unable to concatenate the dataset. Skipping...")

    print("All datasets loaded and combined.")
    print(f"Combined dataset shape: {combined_dataframe.shape}")

    if combined_dataframe.empty:
        raise ValueError("Combined dataframe is empty after processing all datasets.")

    if missing_columns_summary:
        print("Summary of Missing Columns:")
        for file, cols in missing_columns_summary.items():
            print(f"  {file}: {cols}")

    if extra_columns_summary:
        print("Summary of Extra Columns:")
        for file, cols in extra_columns_summary.items():
            print(f"  {file}: {cols}")

    return combined_dataframe

# Function to build neural network
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

# Function to train neural network
def train_neural_network(x_train, y_train, x_test, y_test, num_classes, epochs=20, batch_size=32):
    y_train_encoded = to_categorical(pd.factorize(y_train)[0], num_classes)
    y_test_encoded = to_categorical(pd.factorize(y_test)[0], num_classes)

    model = build_neural_network(input_dim=x_train.shape[1], num_classes=num_classes)

    model.fit(x_train, y_train_encoded,
              validation_data=(x_test, y_test_encoded),
              epochs=epochs,
              batch_size=batch_size,
              verbose=1)

    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test_encoded, axis=1)

    print("Neural Network Evaluation:")
    print("Accuracy:", accuracy_score(y_test_classes, y_pred_classes))
    print("Confusion Matrix:\n", confusion_matrix(y_test_classes, y_pred_classes))
    print("Classification Report:\n", classification_report(y_test_classes, y_pred_classes))

# Main function
def main():
    dataset_directory = '../../data/raw/'
    model_save_path = './random_forest_model.pkl'
    try:
        combined_dataframe = load_multiple_datasets(dataset_directory)
    except ValueError as e:
        print(e)
        return

    try:
        x_train, x_test, y_train, y_test = split_data(combined_dataframe)
    except ValueError as e:
        print(e)
        return

    print("Data preprocessing and splitting completed.")
    print(f"Training features shape: {x_train.shape}")
    print(f"Testing features shape: {x_test.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Testing labels shape: {y_test.shape}")

    x_train_balanced, y_train_balanced = handle_class_imbalance(x_train, y_train)

    # Train Random Forest
    rf_model, rf_params = train_random_forest_with_grid_search(x_train_balanced, y_train_balanced)
    print("Random Forest model trained with optimized hyperparameters.")
    evaluate_model(rf_model, x_test, y_test)

    # Train XGBoost
    xgb_model = train_xgboost(x_train_balanced, y_train_balanced)
    print("XGBoost model trained.")
    evaluate_model(xgb_model, x_test, y_test)

    # Train Stacking
    stacking_model = train_stacking(x_train_balanced, y_train_balanced)
    print("Stacking model trained.")
    evaluate_model(stacking_model, x_test, y_test)

    # Train Voting Classifier
    voting_model = train_voting(x_train_balanced, y_train_balanced)
    print("Voting model trained.")
    evaluate_model(voting_model, x_test, y_test)

    # Train Neural Network
    num_classes = len(y_train.unique())
    train_neural_network(x_train_balanced, y_train_balanced, x_test, y_test, num_classes)

if __name__ == "__main__":
    main()
