import joblib

def save_model(model, file_path):
  joblib.dump(model, file_path)

def load_model(file_path):
  return joblib.load(file_path)