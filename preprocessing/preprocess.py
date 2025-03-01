import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE
from collections import Counter
from preprocessing.features import extract_features

def split_data(df, target_column='eeg_state', test_size=0.2, random_state=42):
    """Splits data into training and testing sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def augment_data(X):
    """Applies noise augmentation to EEG data."""
    noise = np.random.normal(0, 0.05, X.shape)
    return X + noise

def preprocess_data(df, target_column='eeg_state', num_features=10):
    """Applies preprocessing steps including feature extraction, scaling, and balancing."""
    df_features = pd.concat([extract_features(df.iloc[i:i+1]) for i in range(len(df))], ignore_index=True)
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    selector_f = SelectKBest(score_func=f_classif, k=num_features // 2)
    selector_mi = SelectKBest(score_func=mutual_info_classif, k=num_features // 2)
    X_train, X_test, y_train, y_test = split_data(df_features, target_column)
    X_train = imputer.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)
    X_train = augment_data(X_train)
    X_test = imputer.transform(X_test)
    X_test = scaler.transform(X_test)
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("Class distribution after SMOTE:", Counter(y_train))
    X_train_f = selector_f.fit_transform(X_train, y_train)
    X_train_mi = selector_mi.fit_transform(X_train, y_train)
    X_train = np.concatenate((X_train_f, X_train_mi), axis=1)
    X_test_f = selector_f.transform(X_test)
    X_test_mi = selector_mi.transform(X_test)
    X_test = np.concatenate((X_test_f, X_test_mi), axis=1)




    return X_train, X_test, y_train, y_test
