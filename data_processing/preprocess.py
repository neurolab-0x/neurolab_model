import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

def remove_outliers(df):
  for col in df.select_dtypes(include=[np.number]).columns:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
  
  return df

def scale_features(df):
  scaler = StandardScaler()
  numerical_columns = df.select_dtypes(include=[np.number]).columns
  df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
  
  return df

def add_mental_state_labels(df):
  df['mental_state'] = np.random.choice(['Calm', 'Neutral', 'Stressed'], size=len(df))
  return df

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values, scaling, and splitting.

    Args:
        df (pd.DataFrame): Combined dataset.

    Returns:
        tuple: Preprocessed training and testing data (x_train, x_test, y_train, y_test).
    """
    # Check initial state of the dataframe
    print("Initial dataframe shape:", df.shape)
    print("Columns in the dataframe:", df.columns)

    # Separate features and target
    if 'mental_state' not in df.columns:
        raise ValueError("Target column 'mental_state' is missing.")
    
    x = df.drop(columns=['mental_state'], errors='ignore')
    y = df['mental_state']

    print("Features shape after dropping 'mental_state':", x.shape)

    # Handle missing values in numeric columns
    numeric_columns = x.select_dtypes(include=[np.number]).columns

    imputer = SimpleImputer(strategy='mean')
    x[numeric_columns] = imputer.fit_transform(x[numeric_columns])

    # Scale numeric features
    scaler = StandardScaler()
    x[numeric_columns] = scaler.fit_transform(x[numeric_columns])

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print("Training data shape:", x_train.shape)
    print("Testing data shape:", x_test.shape)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    x_train, y_train = smote.fit_resample(x_train, y_train)

    print("Shape after SMOTE:", x_train.shape, y_train.shape)

    return x_train, x_test, y_train, y_test
