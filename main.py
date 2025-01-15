from fastapi import FastAPI
from data_processing.load_data import load_multiple_datasets
from models.random_forest import train_random_forest_with_grid_search
from data_processing.preprocess import preprocess_data
from models.neural_network import train_neural_network
from evaluation.evaluate import evaluate_model
import os

app = FastAPI()

@app.get('/')
def root():
    """
    Welcome endpoint for the API.
    """
    return {"message": "Welcome to the EEG Data Processing API"}

@app.post('/load-data')
def load_data(directory:str):
    """
    Endpoint to load and process datasets from a directory.

    Args:
        directory (str): Path to the directory containing datasets.

    Returns:
        JSON response with the status and shape of the combined dataset.
    """
    if not os.path.exists(directory):
        return {"status": "error", "message": "Directory does not exist"}
    try:
        combined_df = load_multiple_datasets(directory)
        return {"status": "success", "shape": combined_df.shape}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post('/train-random-forest')
def train_rf(directory: str):
    """
    Endpoint to train and evaluate a Random Forest model.

    Args:
        directory (str): Path to the directory containing datasets.

    Returns:
        JSON response with the status, best parameters, and evaluation results.
    """
    try:
        combined_df = load_multiple_datasets(directory)
        x_train, x_test, y_train, y_test = preprocess_data(combined_df)
        rf_model, rf_params = train_random_forest_with_grid_search(x_train, y_train)
        evaluation = evaluate_model(rf_model, x_test, y_test)
        return {
            "status": "success",
            "best_params": rf_params,
            "evaluation": evaluation
        }
    except ValueError as e:
        return { "status" : "error", "message" : str(e) }
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error : {str(e)}" }

@app.post('/train-neural-network')
def train_nn(directory: str):
    """
    Endpoint to train and evaluate a Neural Network model.

    Args:
        directory (str): Path to the directory containing datasets.

    Returns:
        JSON response with the status and evaluation results.
    """
    try:
        combined_df = load_multiple_datasets(directory)
        x_train, x_test, y_train, y_test = preprocess_data(combined_df)
        evaluation = train_neural_network(x_train, y_train, x_test, y_test, num_classes=3)
        return {"status": "success", "evaluation": evaluation}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
