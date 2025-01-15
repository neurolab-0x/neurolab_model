from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd

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

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  return model

def train_neural_network(x_train, y_train, x_test, y_test, num_classes, epochs=20, batch_size=32):
  y_train_encoded = to_categorical(pd.factorize(y_train)[0], num_classes)
  y_test_encoded = to_categorical(pd.factorize(y_test)[0], num_classes)

  model = build_neural_network(input_dim=x_train.shape[1], num_classes=num_classes)
  model.fit(x_train, y_train_encoded, validation_data=(x_test, y_test_encoded), epochs=epochs, batch_size=batch_size, verbose=1)

  y_pred = model.predict(x_test)
  y_pred_classes = np.argmax(y_pred, axis=1)
  y_test_classes = np.argmax(y_test_encoded, axis=1)

  return { "accuracy" : accuracy_score(y_test_classes, y_pred_classes),
          "classification_report" : classification_report(y_test_classes, y_pred_classes),
          "confusion_matrix" : confusion_matrix(y_test_classes, y_pred_classes).tolist() }