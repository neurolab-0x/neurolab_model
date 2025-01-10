import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

file = './data/features_raw.csv'
raw_dataframe = pd.read_csv(file)

dataframe = raw_dataframe.drop(columns=['Unnamed: 32']).fillna(raw_dataframe.mean())

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

print(dataframe[['mental_state'] + list(eeg_columns)].head())


x = dataframe.drop(columns=['mental_state'])
y = dataframe['mental_state']

print("Features shape (X) : ", x.shape)
print("Labels shape (Y) : ", y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("Training feature (X) : ", x_train.shape)
print("Testing feature (X) : ", x_test.shape)
print("Training label (Y) : ", y_train.shape)
print("Testing label (Y) : ", y_test.shape)

rf_classifier = RandomForestClassifier(random_state=42)

rf_classifier.fit(x_train, y_train)

print("Random Forest classier trained successfully")

## Model evaluation

y_pred = rf_classifier.predict(x_test) # Predicting on Testing Data
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuarcy score : {accuracy * 100} %")

#Confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix : {conf_matrix}")

# Classification report
class_report = classification_report(y_test, y_pred)
print(f"Classification report : {class_report}")

param_grid = {
    'n_estimators' : [50, 100, 200],
    'max_depth' : [10, 20, None],
    'min_samples_split' : [2, 5, 10]
}

grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, n_jobs=1, verbose=2)

grid_search.fit(x_train, y_train)

print("Best Hyperparamaters : ", grid_search.best_params_)
print(f"Best score : {grid_search.best_score_:.2f}")

best_params = grid_search.best_params_

optimized_rf_classifier = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    random_state=42
)

optimized_rf_classifier.fit(x_train, y_train)

print("Model trained with optimized Hyperparameters")

# Re-Evaluate Trained Model

y_pred_optimized = optimized_rf_classifier.predict(x_test)
accuaracy_optimized = accuracy_score(y_test, y_pred_optimized)
print(f"Optimized accuaracy : {accuaracy_optimized:.2f}")

conf_matrix_optimized = confusion_matrix(y_test, y_pred_optimized)

print(f"Optimized confusion matrix : {conf_matrix_optimized}")

#Generate classification report

class_report_optimized = classification_report(y_test, y_pred_optimized)
print(f"Optimized Model Classification Report : {class_report_optimized}")

# Feature importance analysis

feature_importances = optimized_rf_classifier.feature_importances_

eeg_columns = x.columns

indices = np.argsort(feature_importances)[::-1]

# Visualizing feature importances

plt.figure(figsize=(10, 6))
plt.title("Feature Importances - EEG Channels")
plt.bar(range(x.shape[1]), feature_importances[indices], align='center')
plt.xticks(range(x.shape[1]), eeg_columns[indices], rotation=90)
plt.xlabel("EEG Channels")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()