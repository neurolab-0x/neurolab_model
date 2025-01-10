import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def load_and_clean_data(file_path):
  raw_dataframe = pd.read_csv(file_path)
  dataframe = raw_dataframe.drop(columns=['Unnamed: 32'], errors='ignore')
  dataframe = dataframe.fillna(dataframe.mean())

  return dataframe

def add_mental_state_labels(dataframe):
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

def split_data(dataframe):
  x = dataframe.drop(columns=['mental_state'])
  y = dataframe['mental_state']

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  return x_train, x_test, y_train, y_test

def train_random_forest(x_train, y_train):
  rf_classifier = RandomForestClassifier(random_state=42)
  rf_classifier.fit(x_train, y_train)

  return rf_classifier

def evaluate_model(model, x_test, y_test):
  y_pred = model.predict(x_test)
  accuracy = accuracy_score(y_test, y_pred)
  conf_matrix = confusion_matrix(y_test, y_pred)
  class_report = classification_report(y_test, y_pred)

  print(f"Accuracy : {accuracy*100:.2f}%")
  print(f"Confusion matrix : \n{conf_matrix}")
  print(f"Classification report : \n{class_report}")

def plot_feature_importance(model, feature_names):

  importances = model.feature_importances_
  indices = np.argsort(importances)[::-1]

  plt.figure(figsize=(10,6))
  plt.title("Feature Importances - EEG Channels")
  plt.bar(range(len(importances)), importances[indices], align='center')
  plt.xticks(range(len(importances)), [feature_names[i] for i in indices])
  plt.xlabel("EEG Channels")
  plt.ylabel("Importance Score")
  plt.tight_layout()
  plt.show()


def main():

  file_path = '../../data/raw/features_raw.csv'

  dataframe = load_and_clean_data(file_path)

  dataframe = add_mental_state_labels(dataframe)

  x_train, x_test, y_train, y_test = split_data(dataframe)

  print("Data preprocessing and splitting completed.")
  print(f"Training feature shape : {x_train.shape}")
  print(f"Testing features shape : {x_test.shape}")
  print(f"Training label shape : {y_train.shape}")
  print(f"Testing label shape : {y_test.shape}")

  model = train_random_forest(x_train, y_train)
  print("Random forest model trained successfully.")

  evaluate_model(model, x_test, y_test)

  feature_names = x_train.columns.tolist()
  plot_feature_importance(model, feature_names)

if __name__ == "__main__":
  main()