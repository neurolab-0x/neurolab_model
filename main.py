from datetime import datetime
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Query, Body, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np
import os
import logging
import uvicorn
from typing import Optional, Dict, Any, List
import base64
import pandas as pd

# Configure logging first
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neurolab_app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("NeuroLabAPI")
logger.setLevel(logging.DEBUG)

# Initialize variables
TENSORFLOW_AVAILABLE = False
model = None
data_handler = None
ml_processing = None

# Try to import core modules first
try:
    from core.data.handler import DataHandler
    data_handler = DataHandler(buffer_size=1000)
    logger.info("Core data handler loaded successfully")
except ImportError as e:
    logger.error(f"Failed to import core data handler: {str(e)}")
    raise

# Try to import ML processing modules
try:
    from core.ml.processing import (
        load_data,
        label_eeg_states,
        extract_features,
        preprocess_data,
        temporal_smoothing,
        calculate_state_durations,
        generate_recommendations
    )
    ml_processing = {
        'load_data': load_data,
        'label_eeg_states': label_eeg_states,
        'extract_features': extract_features,
        'preprocess_data': preprocess_data,
        'temporal_smoothing': temporal_smoothing,
        'calculate_state_durations': calculate_state_durations,
        'generate_recommendations': generate_recommendations
    }
    logger.info("ML processing modules loaded successfully")
except ImportError as e:
    logger.warning(f"ML processing modules import failed: {str(e)}")
    ml_processing = None

# Try to import TensorFlow after core modules
try:
    import tensorflow as tf
    from core.ml.model import load_calibrated_model
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow loaded successfully")
except ImportError as e:
    logger.warning(f"TensorFlow import failed: {str(e)}")
    TENSORFLOW_AVAILABLE = False

ALLOWED_EXTENSIONS = {'.edf', '.bdf', '.gdf', '.csv'}
MAX_FILE_SIZE = 500 * 1024 * 1024
MODEL_PATH = os.getenv("MODEL_PATH", "./processed/trained_model.h5")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan management for the application"""
    global model
    
    if TENSORFLOW_AVAILABLE and ml_processing is not None:
        try:
            logger.info(f"Initializing model from {MODEL_PATH}")
            if os.path.exists(MODEL_PATH):
                model = load_calibrated_model(MODEL_PATH)
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                dummy_input = np.zeros((1, *model.input_shape[1:]))
                _ = model.predict(dummy_input)
                logger.info("Model loaded and warmed up")
            else:
                logger.warning(f"Model file not found at {MODEL_PATH}. Running in minimal mode.")
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            model = None
    else:
        logger.warning("TensorFlow or ML processing not available. Running in minimal mode.")
    
    yield
    logger.info("Application shutdown initiated")

app = FastAPI(
    title="NeuroLab EEG Analysis API",
    description="API for EEG signal processing and mental state classification",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
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

async def save_uploaded_file(file: UploadFile, user_id: str = "anonymous") -> str:
    """Save uploaded file with timestamp prefixing"""
    try:
        os.makedirs("temp", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        safe_filename = ''.join(c for c in file.filename if c.isalnum() or c in '._-')
        file_location = f"temp/{timestamp}_{user_id}_{safe_filename}"
        
        with open(file_location, "wb") as f:
            while content := await file.read(1024 * 1024):
                f.write(content)
                
        if os.path.getsize(file_location) == 0:
            os.remove(file_location)
            raise HTTPException(status_code=400, detail="Empty file uploaded")
            
        return file_location
    except Exception as e:
        logger.error(f"File handling failure: {str(e)}")
        raise HTTPException(500, "File processing error") from e

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        status = {
            "status": "ok",
            "model_loaded": model is not None,
            "tensorflow_available": TENSORFLOW_AVAILABLE,
            "model_path": MODEL_PATH,
            "model_version": "1.0.0",
            "system_time": datetime.now().isoformat()
        }
        
        if model is not None:
            dummy_input = np.zeros((1, *model.input_shape[1:]))
            start_time = datetime.now()
            _ = model.predict(dummy_input)
            latency = (datetime.now() - start_time).total_seconds() * 1000
            status["inference_latency_ms"] = round(latency, 2)
        
        return {
            "status": status,
            "diagnostics": {
                "model_loaded": model is not None,
                "tensorflow_available": TENSORFLOW_AVAILABLE
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.get("/")
async def root():
    """API root with basic information"""
    return {
        "name": "NeuroLab EEG Analysis API",
        "version": "1.0.0",
        "status": "online",
        "tensorflow_available": TENSORFLOW_AVAILABLE,
        "model_loaded": model is not None,
        "documentation": "/docs",
        "timestamp": datetime.now().isoformat()
    }

@app.post('/upload', summary="Advanced EEG analysis", response_description="Cognitive state report", tags=["Analysis"])
async def process_uploaded_file(
    file: Optional[UploadFile] = File(None),
    json_data: Optional[Dict] = Body(None),
    encrypt_response: bool = Query(False, description="Whether to encrypt the response")
):
    """Process uploaded EEG file or JSON data"""
    try:
        logger.info("Processing upload request")
        
        if file is not None:
            # Handle file upload
            logger.info(f"Processing file: {file.filename}")
            validate_file(file)
            file_location = await save_uploaded_file(file, "test_user")
            
            # Load and process data using the data handler
            data_points = data_handler.load_manual_data(
                file_location,
                subject_id="test_user",
                session_id=datetime.now().strftime("%Y%m%d%H%M%S")
            )
        elif json_data is not None:
            # Handle JSON data
            logger.info("Processing JSON data")
            try:
                # Create a temporary file to store the JSON data
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                file_location = f"temp/temp_{timestamp}.csv"
                os.makedirs("temp", exist_ok=True)
                
                # Convert JSON data to DataFrame and save as CSV
                df = pd.DataFrame([json_data])
                df.to_csv(file_location, index=False)
                
                # Load and process data using the data handler
                data_points = data_handler.load_manual_data(
                    file_location,
                    subject_id="test_user",
                    session_id=timestamp
                )
            except Exception as e:
                logger.error(f"Error processing JSON data: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid JSON data format: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="No file or JSON data provided")
        
        if not TENSORFLOW_AVAILABLE or model is None or ml_processing is None:
            # Return basic analysis without model predictions
            session_summary = data_handler.get_session_summary()
            return {
                "status": "success",
                "message": "Basic analysis completed (model not available)",
                "data": session_summary,
                "processing_metadata": {
                    "model_available": False,
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        # Process data with model
        raw_data = ml_processing['load_data'](file_location)
        labeled_data = ml_processing['label_eeg_states'](raw_data)
        features = ml_processing['extract_features'](labeled_data)
        
        X, _, _, _ = ml_processing['preprocess_data'](features)
        
        # Model prediction pipeline
        raw_predictions = model.predict(X.reshape(-1, X.shape[1], 1))
        confidence = np.max(raw_predictions, axis=1) * 100
        state_predictions = np.argmax(raw_predictions, axis=1)
        
        # Post-processing
        smoothed_states = ml_processing['temporal_smoothing'](state_predictions)
        state_durations = ml_processing['calculate_state_durations'](smoothed_states)
        total_duration = sum(state_durations.values()) or 0.1
        
        # Generate recommendations
        recommendations = ml_processing['generate_recommendations'](
            state_durations, 
            total_duration, 
            confidence=np.mean(confidence)
        )
        
        result = {
            "temporal_analysis": {
                "total_duration_sec": round(total_duration, 1),
                "state_distribution": {
                    str(state): f"{duration/total_duration:.1%}"
                    for state, duration in state_durations.items()
                }
            },
            "cognitive_metrics": {
                "mean_confidence": f"{np.mean(confidence):.1f}%",
                "state_transitions": sum(1 for i in range(1, len(smoothed_states)) 
                                       if smoothed_states[i] != smoothed_states[i-1]),
                "dominant_state": int(max(state_durations, key=state_durations.get))
            },
            "clinical_recommendations": recommendations,
            "processing_metadata": {
                "model_version": "1.0.0",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return result
        
    except Exception as e:
        print("Pipeline failure: ", str(e))
        logger.error(f"Pipeline failure: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content = { "message": "Analysis pipeline error", "error": str(e) }
        )
    finally:
        if 'file_location' in locals():
            try:
                os.remove(file_location)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {str(e)}")

def main():
    """Main entry point for the application"""
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="debug"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise

if __name__ == "__main__":
    main()
