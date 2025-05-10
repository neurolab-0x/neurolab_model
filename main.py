from datetime import datetime
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Query, Body, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import numpy as np
import os
import logging
import uvicorn
from typing import Optional, Dict, Any, List
import base64

from api.real_time import process_realtime_data
from models.model import evaluate_model, train_hybrid_model, model_comparison
from preprocessing.features import extract_features
from preprocessing.preprocess import preprocess_data
from preprocessing.load_data import load_data
from preprocessing.labeling import label_eeg_states

from utils.model_loading import load_calibrated_model, get_available_models
from utils.temporal_processing import temporal_smoothing
from utils.duration_calculation import calculate_state_durations
from utils.recommendations import generate_recommendations
from utils.security import DataEncryption, validate_eeg_data, sanitize_model_type
from utils.interpretability import ModelInterpretability
from api.security import rate_limit_middleware, require_user_role, require_admin_role, get_current_user
from api.auth import router as auth_router
from api.streaming_endpoint import router as streaming_router
from config.settings import PROCESSING_CONFIG, MODEL_NAME, MODEL_VERSION, SECURITY_CONFIG, LOGGING_CONFIG

# Configure logging based on settings
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG.get('log_level', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGGING_CONFIG.get('application_log_path', 'neurolab_app.log')),
        logging.StreamHandler()
    ]
)

# Configure security logging
security_logger = logging.getLogger('security')
security_logger.setLevel(getattr(logging, LOGGING_CONFIG.get('log_level', 'INFO')))
security_logger.addHandler(logging.FileHandler(LOGGING_CONFIG.get('security_log_path', 'security.log')))

logger = logging.getLogger("NeuroLabAPI")

ALLOWED_EXTENSIONS = {'.edf', '.bdf', '.gdf', '.csv'}
MAX_FILE_SIZE = 500 * 1024 * 1024
MODEL_PATH = os.getenv("MODEL_PATH", "./processed/trained_model.h5")

# Initialize data encryption
encryption = DataEncryption()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan management for calibrated model loading"""
    global model
    try:
        logger.info(f"Initializing calibrated model from {MODEL_PATH}")
        model = load_calibrated_model(MODEL_PATH)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        dummy_input = np.zeros((1, *model.input_shape[1:]))
        _ = model.predict(dummy_input)
        logger.info("Model loaded and warmed up with temperature scaling")
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        raise RuntimeError("Critical model loading error") from e
    yield
    logger.info("Application shutdown initiated")

app = FastAPI(
    title="NeuroLab EEG Analysis API",
    description="API for EEG signal processing and mental state classification",
    version=MODEL_VERSION,
    lifespan=lifespan
)

# Add middleware for rate limiting
if SECURITY_CONFIG.get('enable_rate_limiting', True):
    @app.middleware("http")
    async def rate_limiting_middleware(request: Request, call_next):
        return await rate_limit_middleware(request, call_next)

# Add CORS middleware with secure configuration
app.add_middleware(
    CORSMiddleware,
    # In production, replace with specific origins
    allow_origins=["*"] if not SECURITY_CONFIG.get('production_mode', False) else [
        "https://your-frontend-domain.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE", "PATCH", "PUT"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# Include routers for authentication and streaming
app.include_router(auth_router)
app.include_router(streaming_router)

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

    # Security check: validate file extension against actual content
    # This helps prevent file type spoofing
    extension = os.path.splitext(file.filename.lower())[1]
    if extension == '.csv':
        # Check first few bytes for CSV format
        content_start = file.file.read(1024)
        file.file.seek(0)  # Reset file pointer
        
        # Basic check for CSV format (comma-separated values)
        if b',' not in content_start and b';' not in content_start:
            logger.warning(f"File claims to be CSV but content doesn't match: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail="File content doesn't match its extension"
            )

async def save_uploaded_file(file: UploadFile, user_id: str = "anonymous") -> str:
    """Secure file handling with timestamp prefixing"""
    try:
        os.makedirs("temp", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Sanitize filename and add user identification for security
        safe_filename = ''.join(c for c in file.filename if c.isalnum() or c in '._-')
        file_location = f"temp/{timestamp}_{user_id}_{safe_filename}"
        
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

@app.post('/upload', summary="Advanced EEG analysis", response_description="Cognitive state report", tags=["Analysis"])
async def process_uploaded_file(
    file: UploadFile = File(...),
    encrypt_response: bool = Query(False, description="Whether to encrypt the response"),
    current_user: Dict = Depends(require_user_role)
):
    """
    Complete processing pipeline with:
    - Temporal pattern analysis
    - Confidence-calibrated predictions
    - Clinical recommendation engine
    - State duration metrics
    
    Security features:
    - Requires authentication
    - File content validation
    - Optional response encryption
    """
    try:
        # Log file upload attempt
        security_logger.info(f"User {current_user['sub']} uploading file: {file.filename}")
        
        validate_file(file)
        file_location = await save_uploaded_file(file, current_user['sub'])
        
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
        recommendations = generate_recommendations(state_durations, total_duration, confidence=np.mean(confidence))
        
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
                "smoothing_window": f"{PROCESSING_CONFIG['smoothing_window']} samples",
                "model_version": MODEL_VERSION,
                "timestamp": datetime.now().isoformat(),
                "user_id": current_user['sub']
            }
        }
        
        # Encrypt response if requested
        if encrypt_response or SECURITY_CONFIG.get('encryption_for_transit', False):
            encrypted_result = encryption.encrypt_data(result)
            return JSONResponse(content={
                "encrypted": True,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model_version": MODEL_VERSION,
                    "user_id": current_user['sub']
                },
                "data": base64.b64encode(encrypted_result).decode()
            })
        
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Pipeline failure: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content = { "message": "Analysis pipeline error", "error": str(e) }
        )
    finally:
        if 'file_location' in locals() and os.path.exists(file_location):
            os.remove(file_location)

@app.post("/realtime/", 
          summary="Real-time processing", 
          tags=["Real-time"])
async def realtime_data(
    data: dict, 
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(require_user_role)
):
    """Initialize real-time processing pipeline with authentication"""
    try:
        # Validate input data
        if 'eeg_data' not in data:
            raise HTTPException(status_code=400, detail="Missing 'eeg_data' field")
        
        # Convert to numpy and validate
        eeg_data = np.array(data['eeg_data'])
        if not validate_eeg_data(eeg_data):
            raise HTTPException(status_code=400, detail="Invalid EEG data format")
        
        # Log request
        logger.info(f"Real-time processing request from user {current_user['sub']}")
        
        # Add user info to data for tracking
        data['user_id'] = current_user['sub']
        
        # Start processing
        background_tasks.add_task(process_realtime_data, data, model)
        return {"message": "Real-time analysis started"}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Real-time init error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": "Real-time processing failed", "error": str(e)}
        )

@app.post("/retrain", summary="Model retraining", tags=["Training"])
async def retrain_model(
    file: UploadFile = File(...),
    model_type: str = Query("enhanced_cnn_lstm", 
                          description="Model architecture to use: 'original', 'enhanced_cnn_lstm', 'resnet_lstm', 'transformer'"),
    hyper_params: Dict[str, Any] = Body(
        default={
            "dropout_rate": 0.3,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 30
        },
        description="Hyperparameters for model training"
    ),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: Dict = Depends(require_admin_role)  # Require admin role
):
    """
    Enhanced model retraining endpoint with architecture selection and hyperparameter tuning.
    Runs in the background for better API performance.
    
    Security features:
    - Requires admin role
    - Validates and sanitizes inputs
    - Logs training activities
    """
    try:
        # Log retraining attempt
        security_logger.info(f"Admin {current_user['sub']} initiated model retraining: {model_type}")
        
        # Sanitize model type
        model_type = sanitize_model_type(model_type)
        
        # Validate model type
        valid_model_types = ['original', 'enhanced_cnn_lstm', 'resnet_lstm', 'transformer']
        if model_type not in valid_model_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type. Supported types: {', '.join(valid_model_types)}"
            )
            
        # Validate file
        validate_file(file)
        file_location = await save_uploaded_file(file, current_user['sub'])
        
        # Start training in background
        background_tasks.add_task(run_training, file_location, model_type, hyper_params, current_user['sub'])
        logger.info(f"Retraining started with model_type: {model_type}")
    
        return {
            "message": f"Training {model_type} model initiated",
            "hyperparameters": hyper_params,
            "admin_id": current_user['sub']
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": "Retraining failed", "error": str(e) }
        )

def run_training(file_path: str, model_type: str = "enhanced_cnn_lstm", hyper_params: Dict = None, user_id: str = "system"):
    """Enhanced background training task with model type selection and logging"""
    try:
        if hyper_params is None:
            hyper_params = {}
            
        logger.info(f"Starting {model_type} model retraining cycle by user {user_id}")
        raw_data = load_data(file_path)
        labeled_data = label_eeg_states(raw_data)
        X_train, X_test, y_train, y_test = preprocess_data(labeled_data)
        
        # Train model with selected architecture
        new_model, history = train_hybrid_model(
            X_train, 
            y_train,
            model_type=model_type,
            **hyper_params
        )
        
        # Evaluate on test set
        test_loss, test_accuracy = evaluate_model(new_model, X_test, y_test)
        
        logger.info(f"Model training complete. Test accuracy: {test_accuracy:.2f}")
        
        # Log training result in secure log
        security_logger.info(f"Model {model_type} training completed by {user_id}. "
                            f"Accuracy: {test_accuracy:.2f}, Loss: {test_loss:.4f}")
        
    except Exception as e:
        logger.error(f"Training process error: {str(e)}")
        security_logger.error(f"Training failure for user {user_id}: {str(e)}")
    finally:
        # Always clean up the file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/compare_models", summary="Compare model architectures", tags=["Training"])
async def compare_model_architectures(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: Dict = Depends(require_admin_role)  # Require admin role
):
    """Compare performance of different model architectures with secure file handling"""
    try:
        # Log model comparison attempt
        security_logger.info(f"Admin {current_user['sub']} initiated model comparison")
        
        validate_file(file)
        file_location = await save_uploaded_file(file, current_user['sub'])
        
        # Start comparison in background
        background_tasks.add_task(run_model_comparison, file_location, current_user['sub'])
        
        return {
            "message": "Model architecture comparison started",
            "status": "processing",
            "admin_id": current_user['sub']
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Model comparison error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": "Model comparison failed", "error": str(e)}
        )

def run_model_comparison(file_path: str, user_id: str = "system"):
    """Background task for model architecture comparison with logging"""
    try:
        logger.info(f"Starting model comparison by user {user_id}")
        raw_data = load_data(file_path)
        labeled_data = label_eeg_states(raw_data)
        X_train, X_test, y_train, y_test = preprocess_data(labeled_data)
        
        # Run comparison
        comparison_results = model_comparison(X_train, y_train, X_test, y_test)
        
        # Log results
        for model_name, metrics in comparison_results.items():
            logger.info(f"Model {model_name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
        
        security_logger.info(f"Model comparison completed by {user_id}")
        
    except Exception as e:
        logger.error(f"Model comparison process error: {str(e)}")
    finally:
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/models", summary="List available models", tags=["Models"])
async def list_available_models(current_user: Dict = Depends(get_current_user)):
    """List available trained models with model information"""
    try:
        models = get_available_models()
        return {
            "available_models": models,
            "active_model": MODEL_PATH,
            "user_id": current_user['sub']
        }
    except Exception as e:
        logger.error(f"Model listing error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to list models", "error": str(e)}
        )

@app.post("/set_active_model", summary="Set active model", tags=["Models"])
async def set_active_model(
    model_path: str = Body(..., embed=True),
    current_user: Dict = Depends(require_admin_role)  # Require admin role
):
    """Set the active model for inference with admin role requirement"""
    try:
        # Log model switching attempt
        security_logger.info(f"Admin {current_user['sub']} changing active model to {model_path}")
        
        # Sanitize model path to prevent path traversal
        model_filename = os.path.basename(model_path)
        safe_path = os.path.join("./processed", model_filename)
        
        # Check if model exists
        if not os.path.exists(safe_path):
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {model_filename}"
            )
            
        global model, MODEL_PATH
        # Load new model
        new_model = load_calibrated_model(safe_path)
        
        # Test model with dummy data
        dummy_input = np.zeros((1, *new_model.input_shape[1:]))
        _ = new_model.predict(dummy_input)
        
        # Update model reference
        model = new_model
        MODEL_PATH = safe_path
        
        logger.info(f"Active model switched to {MODEL_PATH}")
        
        return {
            "message": "Model switched successfully",
            "active_model": MODEL_PATH,
            "admin_id": current_user['sub']
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Model switching error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to switch model", "error": str(e)}
        )

@app.get("/health")
async def health_check():
    """System health check and status monitoring"""
    try:
        # Check if model is loaded
        if 'model' not in globals():
            return JSONResponse(
                status_code=503,
                content={"status": "error", "message": "Model not initialized"}
            )
            
        # Test model with dummy data to ensure it's responsive
        dummy_input = np.zeros((1, *model.input_shape[1:]))
        start_time = datetime.now()
        _ = model.predict(dummy_input)
        latency = (datetime.now() - start_time).total_seconds() * 1000
        
        # System status information
        status = {
            "status": "ok",
            "model_loaded": True,
            "model_path": MODEL_PATH,
            "model_version": MODEL_VERSION,
            "inference_latency_ms": round(latency, 2),
            "system_time": datetime.now().isoformat(),
            "security_enabled": SECURITY_CONFIG.get('require_authentication', False)
        }
        
        # Additional diagnostics
        memory_info = {"model_loaded": True}  # Placeholder for memory statistics
        
        return {
            "status": status,
            "diagnostics": memory_info
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
        "version": MODEL_VERSION,
        "model": MODEL_NAME,
        "status": "online",
        "documentation": "/docs",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/interpretability/explain", summary="Get model explanations", tags=["Interpretability"])
async def get_model_explanation(
    file: UploadFile = File(...),
    explanation_type: str = Query("shap", description="Explanation type: 'shap' or 'lime'"),
    num_samples: int = Query(10, description="Number of samples to explain"),
    current_user: Dict = Depends(require_user_role)
):
    """
    Generate model explanations for interpretability
    
    Returns SHAP or LIME explanations for the provided EEG data.
    Helps understand which features contribute most to the model's prediction.
    
    Security:
    - Requires user authentication
    - Validates input file
    - Sanitizes parameters
    """
    try:
        # Log explanation request
        logger.info(f"Model explanation requested by user {current_user['sub']}")
        
        # Validate explanation type
        valid_types = ["shap", "lime", "both"]
        if explanation_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid explanation type. Must be one of: {', '.join(valid_types)}"
            )
        
        # Validate file
        validate_file(file)
        file_location = await save_uploaded_file(file, current_user['sub'])
        
        # Process data
        raw_data = load_data(file_location)
        labeled_data = label_eeg_states(raw_data)
        features = extract_features(labeled_data)
        
        X, _, _, _ = preprocess_data(features)
        
        # Create interpretability handler
        interpreter = ModelInterpretability(model)
        
        # Set feature names
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        interpreter.set_feature_names(feature_names)
        
        results = {}
        
        # Generate explanations based on the requested type
        if explanation_type in ["shap", "both"]:
            # Limit number of samples for SHAP for performance
            shap_samples = min(num_samples, 20)
            shap_results = interpreter.explain_with_shap(
                X[:shap_samples], 
                n_samples=shap_samples
            )
            
            # Remove the explainer object which can't be serialized
            if "explainer" in shap_results:
                del shap_results["explainer"]
                
            # Convert numpy arrays to lists for JSON serialization
            if "feature_importance" in shap_results:
                # Convert feature importance values for each class to list
                for cls, importance in shap_results["feature_importance"].items():
                    if isinstance(importance, np.ndarray):
                        shap_results["feature_importance"][cls] = importance.tolist()
            
            results["shap"] = shap_results
        
        if explanation_type in ["lime", "both"]:
            # Get LIME explanation for first sample
            lime_results = interpreter.explain_with_lime(
                X[:num_samples], 
                sample_idx=0,
                num_features=10
            )
            
            # Remove the explanation object which can't be serialized
            if "explanation" in lime_results:
                # Extract top features and their importance
                top_features = lime_results["feature_importance"]
                lime_results["feature_importance"] = top_features
                del lime_results["explanation"]
                
            results["lime"] = lime_results
        
        # Return the explanations
        return {
            "explanation_type": explanation_type,
            "results": results,
            "model_info": {
                "model_version": MODEL_VERSION,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"message": "Explanation failed", "error": str(e)}
        )
    finally:
        # Clean up the file
        if 'file_location' in locals() and os.path.exists(file_location):
            os.remove(file_location)

@app.post("/interpretability/calibrate", summary="Calibrate model confidence", tags=["Interpretability"])
async def calibrate_model_confidence(
    file: UploadFile = File(...),
    method: str = Query("temperature_scaling", description="Calibration method: 'temperature_scaling', 'platt_scaling', or 'isotonic'"),
    current_user: Dict = Depends(require_admin_role)  # Require admin for calibration
):
    """
    Calibrate model's confidence scores for improved reliability
    
    Uses validation data to calibrate the model's predicted probabilities,
    ensuring they match the true correctness likelihood (reliability).
    
    Security:
    - Requires admin authentication
    - Validates input file
    - Sanitizes parameters
    """
    try:
        # Log calibration request
        security_logger.info(f"Admin {current_user['sub']} requested model confidence calibration")
        
        # Validate method
        valid_methods = ["temperature_scaling", "platt_scaling", "isotonic"]
        if method not in valid_methods:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid calibration method. Must be one of: {', '.join(valid_methods)}"
            )
        
        # Validate file
        validate_file(file)
        file_location = await save_uploaded_file(file, current_user['sub'])
        
        # Process data - here we need labeled data for calibration
        raw_data = load_data(file_location)
        labeled_data = label_eeg_states(raw_data)
        
        # Extract features and preprocess
        features = extract_features(labeled_data)
        X_val, _, y_val, _ = preprocess_data(features)
        
        # Create interpretability handler
        interpreter = ModelInterpretability(model)
        
        # Perform calibration
        calibration_result = interpreter.calibrate_confidence(X_val, y_val, method=method)
        
        # Convert numpy arrays to lists for JSON serialization
        if "expected_calibration_error" in calibration_result:
            ece = calibration_result["expected_calibration_error"]
            calibration_result["expected_calibration_error"] = float(ece)
        
        if "temperature" in calibration_result:
            temp = calibration_result["temperature"]
            calibration_result["temperature"] = float(temp)
        
        # Remove large arrays to reduce response size
        large_keys = ["uncalibrated_predictions", "calibrated_predictions", "calibrators"]
        for key in large_keys:
            if key in calibration_result:
                del calibration_result[key]
        
        # Return the calibration results
        return {
            "calibration_method": method,
            "calibration_info": calibration_result,
            "model_info": {
                "model_version": MODEL_VERSION,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Calibration error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"message": "Calibration failed", "error": str(e)}
        )
    finally:
        # Clean up the file
        if 'file_location' in locals() and os.path.exists(file_location):
            os.remove(file_location)

@app.get("/interpretability/reliability_diagram", summary="Get reliability diagram", tags=["Interpretability"])
async def get_reliability_diagram(
    current_user: Dict = Depends(require_user_role),
):
    """
    Generate and retrieve a reliability diagram (calibration curve)
    
    Returns a base64-encoded image of the model's reliability diagram,
    showing how predicted probabilities match observed frequencies.
    
    Security:
    - Requires user authentication
    """
    try:
        import matplotlib.pyplot as plt
        import io
        import base64
        
        # Create interpretability handler
        interpreter = ModelInterpretability(model)
        
        # Generate synthetic validation data if we don't have any
        # In production, you would use actual validation data
        from tests.perfomance_test import generate_dummy_data
        X_dummy, y_dummy = generate_dummy_data(num_samples=100, sequence_length=10, num_classes=3)
        
        # Get predictions
        preds = model.predict(X_dummy)
        
        # Create reliability diagram
        fig = interpreter.plot_calibration_curve(preds, y_dummy, n_bins=10)
        
        # Convert plot to base64 image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        # Return the image
        return {
            "reliability_diagram": img_str,
            "content_type": "image/png",
            "encoding": "base64",
            "model_info": {
                "model_version": MODEL_VERSION,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    except Exception as e:
        logger.error(f"Reliability diagram error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to generate reliability diagram", "error": str(e)}
        )

if __name__ == "__main__":
    uvicorn.run(
        app="main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        log_config=None
    )
