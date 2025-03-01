from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from api.real_time import process_realtime_data
from models.load_trained_model import load_trained_model
from models.model import evaluate_model, train_hybrid_model
from preprocessing.features import extract_features
from preprocessing.preprocess import preprocess_data
from preprocessing.load_data import load_data
from preprocessing.labeling import label_eeg_states

import os
import uvicorn

app = FastAPI()

model_path="./data/processed/trained_model.h5"

@app.post('/upload')
async def process_uploaded_file(file: UploadFile = File(...)):
    """Handles uploaded EEG file, processes it, and runs model inference."""
    file_location = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)

    with open(file_location, "wb") as f:
        f.write(await file.read())

    print(f"File {file.filename} saved at {file_location}")
    print(f"File size: {os.path.getsize(file_location)} bytes")

    if os.path.getsize(file_location) == 0:
        return {"error": "Uploaded file is empty!"}

    df = load_data(file_location)
    print(f"Dataset shape : {df.shape}")
    df = label_eeg_states(df)

    features_df = extract_features(df)
    print(f"Extracted features shape : {features_df.shape}")
    
    X, _, _, _ = preprocess_data(features_df)

    model_path = "./processed/trained_model.h5"
    print(f"Loading trained model at {model_path} ...")
    model = load_trained_model(model_path)
    print("Predicting eeg states based on uploaded data ...")

    predictions = model.predict(X.reshape(-1, X.shape[1], 1))
    print("Linearising predictions ...")
    predicted_state = predictions.argmax(axis=1)
    confidence_state = predicted_state.max(predictions, axis=1) * 100
    print("Cleaning up memory leakages ...")
    os.remove(file_location)
    recommendations = [{
        "0" : "",
        "1" : "",
        "2" : "",
        "3" : ""
    }]
    print("Process finished")
    return { "predicted_state": predicted_state.tolist(), "confidence_level": confidence_state.tolist() }


@app.post("/realtime/")
async def realtime_data(data: dict, background_tasks: BackgroundTasks):
    """Handles real-time EEG data processing."""
    background_tasks.add_task(process_realtime_data, data, model=load_trained_model(model_path))
    return {"message": "Real-time data processing started"}

from fastapi import File, UploadFile

@app.post("/retrain")
async def retrain_model(file: UploadFile = File(...)):
    """Re-trains the existing model using uploaded dataset."""
    if not file:
        print("No file uploaded")

    file_location = f"./data/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    run_training(file_location)
    return {"message": "Model retraining started"}


def run_training(file_path):
    """Runs model retraining in the background."""
    df = load_data(file_path)
    print(df.describe())
    df = label_eeg_states(df)
    print(df.columns)

    X_train, X_test, y_train, y_test = preprocess_data(df)

    hybrid_model = train_hybrid_model(X_train, y_train)
    evaluate_model(hybrid_model, X_test, y_test)
    return { "Processing, training and evaluation complete." }

@app.get("/")
def root():
    return {"message": "NeuroLab AI API is running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
