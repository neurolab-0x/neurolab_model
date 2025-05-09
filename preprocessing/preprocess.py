import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
from preprocessing.features import extract_features
from utils.artifacts import clean_eeg
from utils.filters import apply_eeg_preprocessing, filter_eeg_bands

def split_data(df, target_column='eeg_state', test_size=0.2, random_state=42, stratify=True):
    """Splits data into training and testing sets with stratification option."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    if stratify:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

def augment_data(X, y, noise_level=0.05, shift_max=5, num_augmented_samples=None):
    """Applies advanced data augmentation to EEG data."""
    if num_augmented_samples is None:
        num_augmented_samples = X.shape[0] // 4  # Add 25% more samples by default
    
    n_samples, n_features = X.shape
    augmented_X = []
    augmented_y = []
    
    for _ in range(num_augmented_samples):
        # Randomly select a sample to augment
        idx = np.random.randint(0, n_samples)
        x = X[idx].copy()
        
        # Apply random noise
        noise = np.random.normal(0, noise_level, n_features)
        x_noise = x + noise
        
        # Apply small time shifts (simulate small alignment errors)
        shift = np.random.randint(-shift_max, shift_max + 1)
        if shift != 0:
            x_shift = np.roll(x, shift)
            # Ensure continuity at edges
            if shift > 0:
                x_shift[:shift] = x_shift[shift]
            else:
                x_shift[shift:] = x_shift[shift-1]
        else:
            x_shift = x
        
        # Combine transformations
        x_augmented = x_shift + (noise * 0.5)  # Reduce noise level for combined
        
        augmented_X.append(x_augmented)
        augmented_y.append(y[idx])
    
    # Combine original and augmented data
    augmented_X = np.vstack([X, np.array(augmented_X)])
    augmented_y = np.concatenate([y, np.array(augmented_y)])
    
    return augmented_X, augmented_y

def remove_outliers(X, y, contamination=0.05):
    """Remove outliers using isolation forest."""
    from sklearn.ensemble import IsolationForest
    
    # Train isolation forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_pred = iso_forest.fit_predict(X)
    
    # Keep only inliers
    mask = outlier_pred == 1
    X_clean = X[mask]
    y_clean = y[mask]
    
    print(f"Removed {len(y) - len(y_clean)} outliers ({(1 - len(y_clean)/len(y))*100:.2f}%)")
    
    return X_clean, y_clean

def preprocess_data(df, target_column='eeg_state', num_features=None, clean_artifacts=True, 
                    use_robust_scaler=False, balance_method='smote'):
    """
    Enhanced preprocessing pipeline with artifact cleaning, advanced scaling options,
    and choice of balancing method.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Input dataframe containing EEG data and state labels
    target_column : str
        Column name for the target variable
    num_features : int or None
        Number of features to select (if None, uses all features)
    clean_artifacts : bool
        Whether to apply artifact cleaning
    use_robust_scaler : bool
        Whether to use RobustScaler instead of StandardScaler
    balance_method : str
        Method to balance classes ('smote', 'adasyn', or None)
        
    Returns:
    -------
    X_train, X_test, y_train, y_test
    """
    print("Starting enhanced preprocessing pipeline...")
    
    # Extract features if not already done
    if 'eeg_state' in df.columns and len(df.columns) < 10:
        print("Extracting features from raw data...")
        df_features = pd.DataFrame()
        for i in range(len(df)):
            # Extract single row
            row_df = df.iloc[i:i+1]
            
            # Apply preprocessing to raw signals if needed
            if clean_artifacts:
                for col in row_df.columns:
                    if col != target_column:
                        # Get the signal
                        signal = np.array(row_df[col])
                        
                        if len(signal) > 10:  # Only process if enough samples
                            # Apply artifact cleaning
                            cleaned_signal, _ = clean_eeg(signal.reshape(1, -1))
                            
                            # Apply EEG preprocessing (filtering)
                            cleaned_signal = apply_eeg_preprocessing(cleaned_signal[0])
                            
                            # Update the DataFrame
                            row_df[col] = cleaned_signal
            
            # Extract features from (possibly cleaned) signals
            features = extract_features(row_df)
            df_features = pd.concat([df_features, features], ignore_index=True)
    else:
        print("Using provided feature dataframe...")
        df_features = df.copy()
    
    print(f"Feature extraction complete. Shape: {df_features.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(df_features, target_column, stratify=True)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Fill missing values
    print("Handling missing values...")
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    # Scale the features
    print("Scaling features...")
    if use_robust_scaler:
        scaler = RobustScaler()  # More robust to outliers
    else:
        scaler = StandardScaler()
        
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Augment training data
    print("Augmenting training data...")
    X_train, y_train = augment_data(X_train, y_train)
    print(f"After augmentation - Training set: {X_train.shape}")
    
    # Balance classes if requested
    if balance_method:
        print(f"Balancing classes using {balance_method}...")
        print("Class distribution before balancing:", Counter(y_train))
        
        if balance_method.lower() == 'smote':
            balancer = SMOTE(sampling_strategy='auto', random_state=42)
        elif balance_method.lower() == 'adasyn':
            balancer = ADASYN(sampling_strategy='auto', random_state=42)
        else:
            raise ValueError(f"Unknown balance method: {balance_method}")
            
        X_train, y_train = balancer.fit_resample(X_train, y_train)
        print("Class distribution after balancing:", Counter(y_train))
    
    # Feature selection if requested
    if num_features is not None and num_features < X_train.shape[1]:
        print(f"Selecting top {num_features} features...")
        
        # Use a combination of multiple feature selection methods for robustness
        selector_f = SelectKBest(score_func=f_classif, k=num_features // 2)
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=num_features // 2)
        
        X_train_f = selector_f.fit_transform(X_train, y_train)
        X_train_mi = selector_mi.fit_transform(X_train, y_train)
        
        X_test_f = selector_f.transform(X_test)
        X_test_mi = selector_mi.transform(X_test)
        
        X_train = np.concatenate((X_train_f, X_train_mi), axis=1)
        X_test = np.concatenate((X_test_f, X_test_mi), axis=1)
        print(f"After feature selection - Feature count: {X_train.shape[1]}")
    
    print("Preprocessing complete!")
    return X_train, X_test, y_train, y_test
