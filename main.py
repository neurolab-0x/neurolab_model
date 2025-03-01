from datetime import datetime
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np
import os
import logging
import uvicorn

# Core components
from api.real_time import process_realtime_data
from models.model import evaluate_model, train_hybrid_model
from preprocessing.features import extract_features
from preprocessing.preprocess import preprocess_data
from preprocessing.load_data import load_data
from preprocessing.labeling import label_eeg_states

# Utility imports
from utils.model_loading import load_calibrated_model
from utils.temporal_processing import temporal_smoothing
from utils.duration_calculation import calculate_state_durations
from utils.recommendations import generate_recommendations
from config.settings import PROCESSING_CONFIG, THRESHOLDS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NeuroLabAPI")

# Constants
ALLOWED_EXTENSIONS = {'.edf', '.bdf', '.gdf', '.csv'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
MODEL_PATH = os.getenv("MODEL_PATH", "./processed/trained_model.h5")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan management for calibrated model loading"""
    global model
    try:
        logger.info(f"Initializing calibrated model from {MODEL_PATH}")
        model = load_calibrated_model(MODEL_PATH)
        logger.info("Model loaded with temperature scaling")
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        raise RuntimeError("Critical model loading error") from e
    yield
    logger.info("Application shutdown initiated")

app = FastAPI(
    title="NeuroLab AI API",
    description="Modular EEG Processing Platform",
    version="2.1.1",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_file(file: UploadFile):
    """Validate uploaded file parameters"""
    if not file.filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds size limit of {MAX_FILE_SIZE//1024//1024}MB"
        )

async def save_uploaded_file(file: UploadFile) -> str:
    """Secure file handling with timestamp prefixing"""
    try:
        os.makedirs("temp", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_location = f"temp/{timestamp}_{file.filename}"
        
        with open(file_location, "wb") as f:
            while content := await file.read(1024 * 1024):  # 1MB chunks
                f.write(content)
                
        if os.path.getsize(file_location) == 0:
            os.remove(file_location)
            raise HTTPException(status_code=400, detail="Empty file uploaded")
            
        return file_location
    except Exception as e:
        logger.error(f"File handling failure: {str(e)}")
        raise HTTPException(500, "File processing error") from e

@app.post('/upload', 
          summary="Enhanced EEG analysis", 
          response_description="Cognitive state report",
          tags=["Analysis"])
async def process_uploaded_file(file: UploadFile = File(...)):
    """
    Complete processing pipeline with:
    - Temporal pattern analysis
    - Confidence-calibrated predictions
    - Clinical recommendation engine
    - State duration metrics
    """
    try:
        validate_file(file)
        file_location = await save_uploaded_file(file)
        
        logger.info(f"Processing dataset: {file.filename}")
        raw_data = load_data(file_location)
        labeled_data = label_eeg_states(raw_data)
        features = extract_features(labeled_data)
        
        X, _, _, _ = preprocess_data(features)
        
        # Model prediction pipeline
        raw_predictions = model.predict(X.reshape(-1, X.shape[1], 1))
        confidence = np.max(raw_predictions, axis=1) * 100
        state_predictions = np.argmax(raw_predictions, axis=1)
        
        # Post-processing
        smoothed_states = temporal_smoothing(state_predictions)
        state_durations = calculate_state_durations(smoothed_states)
        total_duration = sum(state_durations.values()) or 0.1  # Prevent zero division
        
        # Generate clinical insights
        recommendations = generate_recommendations(
            state_durations,
            total_duration,
            confidence=np.mean(confidence)
        )
        
        return {
            "temporal_analysis": {
                "total_duration_sec": round(total_duration, 1),
                "state_distribution": {
                    state: f"{duration/total_duration:.1%}"
                    for state, duration in state_durations.items()
                }
            },
            "cognitive_metrics": {
                "mean_confidence": f"{np.mean(confidence):.1f}%",
                "state_transitions": sum(1 for i in range(1, len(smoothed_states)) 
                                       if smoothed_states[i] != smoothed_states[i-1]),
                "dominant_state": max(state_durations, key=state_durations.get)
            },
            "clinical_recommendations": recommendations,
            "processing_metadata": {
                "smoothing_window": f"{PROCESSING_CONFIG['smoothing_window']} samples",
                "model_version": "2.1-calibrated",
                "analysis_date": datetime.now().isoformat()
            }
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Pipeline failure: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"message": "Analysis pipeline error"}
        )
    finally:
        if 'file_location' in locals() and os.path.exists(file_location):
            os.remove(file_location)

@app.post("/realtime/", 
          summary="Real-time processing", 
          tags=["Real-time"])
async def realtime_data(data: dict, background_tasks: BackgroundTasks):
    """Initialize real-time processing pipeline"""
    try:
        background_tasks.add_task(process_realtime_data, data, model)
        return {"message": "Real-time analysis started"}
    except Exception as e:
        logger.error(f"Real-time init error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": "Real-time processing failed"}
        )

@app.post("/retrain", 
          summary="Model retraining", 
          tags=["Training"])
async def retrain_model(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Model retraining endpoint"""
    try:
        validate_file(file)
        file_location = await save_uploaded_file(file)
        
        background_tasks.add_task(run_training, file_location)
        return {"message": "Retraining job initiated"}
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": "Retraining failed"}
        )

def run_training(file_path: str):
    """Background training task"""
    try:
        logger.info("Starting model retraining cycle")
        raw_data = load_data(file_path)
        labeled_data = label_eeg_states(raw_data)
        X_train, X_test, y_train, y_test = preprocess_data(labeled_data)
        
        new_model = train_hybrid_model(X_train, y_train)
        metrics = evaluate_model(new_model, X_test, y_test)
        
        logger.info(f"Training complete: {metrics}")
        
    except Exception as e:
        logger.error(f"Training failure: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/health", 
         summary="System health check", 
         tags=["Monitoring"])
async def health_check():
    """Service health status endpoint"""
    return {
        "status": "OPERATIONAL",
        "timestamp": datetime.now().isoformat(),
        "model_status": "CALIBRATED" if model else "ERROR",
        "active_features": list(PROCESSING_CONFIG.keys())
    }

@app.get("/", 
         summary="Service information", 
         tags=["Info"])
def root():
    """API root endpoint"""
    return {
        "service": "NeuroLab AI",
        "version": "2.1.1",
        "modules": [
            "Temporal Analysis",
            "Clinical Recommendations",
            "Calibrated Predictions"
        ],
        "documentation": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        app="main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        log_config=None
    )
