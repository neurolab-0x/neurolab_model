from datetime import datetime
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Query, Body, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import uvicorn
from typing import Optional, Dict, Any, List
import base64
import pandas as pd

# Import utility modules
from utils.file_handler import validate_file, save_uploaded_file
from utils.model_manager import ModelManager
from utils.ml_processor import MLProcessor

# Configure logging
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

# Initialize components
model_manager = ModelManager()
ml_processor = MLProcessor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan management for the application"""
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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": model_manager.get_health_status(),
            "diagnostics": {
                "model_loaded": model_manager.model is not None,
                "tensorflow_available": model_manager.tensorflow_available
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
        "description": "API for EEG signal processing and mental state classification",
        "endpoints": {
            "health": "/health",
            "upload": "/upload",
            "analyze": "/analyze",
            "calibrate": "/calibrate",
            "recommendations": "/recommendations"
        }
    }

@app.post('/upload', summary="Advanced EEG analysis", response_description="Cognitive state report", tags=["Analysis"])
async def process_uploaded_file(
    file: Optional[UploadFile] = File(None),
    json_data: Optional[Dict] = Body(None),
    encrypt_response: bool = Query(False, description="Whether to encrypt the response")
):
    """Process uploaded EEG file or JSON data"""
    try:
        if file:
            validate_file(file)
            file_location = await save_uploaded_file(file)
            result = ml_processor.process_eeg_data(file_location, "anonymous", "session_1")
        elif json_data:
            result = ml_processor.process_eeg_data(json_data, "anonymous", "session_1")
        else:
            raise HTTPException(status_code=400, detail="No file or data provided")
            
        if encrypt_response:
            result = base64.b64encode(str(result).encode()).decode()
            
        return result
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/analyze', summary="Analyze EEG data", response_description="Analysis results", tags=["Analysis"])
async def analyze_eeg_data(
    data: Dict[str, Any] = Body(..., description="EEG data to analyze"),
    background_tasks: BackgroundTasks = None
):
    """Analyze EEG data and return results"""
    try:
        result = ml_processor.process_eeg_data(
            data,
            subject_id=data.get('subject_id', 'anonymous'),
            session_id=data.get('session_id', 'session_1')
        )
        return result
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/calibrate', summary="Calibrate model", response_description="Calibration results", tags=["Model"])
async def calibrate_model(
    calibration_data: Dict[str, Any] = Body(..., description="Calibration data"),
    background_tasks: BackgroundTasks = None
):
    """Calibrate the model with new data"""
    try:
        if not model_manager.model:
            raise HTTPException(status_code=503, detail="Model not available")
            
        # Add calibration logic here
        return {"status": "calibration_started", "message": "Calibration process initiated"}
    except Exception as e:
        logger.error(f"Error calibrating model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/recommendations', summary="Get recommendations", response_description="Recommendations based on analysis", tags=["Analysis"])
async def get_recommendations(
    session_id: str = Query(..., description="Session ID to get recommendations for"),
    subject_id: str = Query(..., description="Subject ID")
):
    """Get recommendations based on previous analysis"""
    try:
        # Add recommendation logic here
        return {
            "session_id": session_id,
            "subject_id": subject_id,
            "recommendations": []
        }
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Main entry point"""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
