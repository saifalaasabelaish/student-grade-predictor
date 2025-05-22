"""
FastAPI backend for student grade prediction.
"""
import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Dict, Any, List
import logging

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import load_model, NaiveBayesClassifier
from schemas import StudentFeatures, PredictionResponse, ErrorResponse, HealthResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model
model: NaiveBayesClassifier = None
model_metadata: Dict[str, Any] = None

# FastAPI app
app = FastAPI(
    title="Student Grade Prediction API",
    description="API for predicting student grades using Naive Bayes classifier",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Manual encoding maps (must match training encoding)
ENCODING_MAPS = {
    'Gender': {'female': 0, 'male': 1},
    'Transportation': {'Bus': 0, 'Car': 1},
    'Accommodation': {'Dorms': 0, 'With familly': 1},
    'MidExam': {
        'Closest day to the exam': 0,
        'Regularly during the semester': 1
    },
    'TakingNotes': {'Always': 0, 'Sometimes': 1}
}

# CRITICAL FIX: Grade decoding map (from encoded integers back to letter grades)
GRADE_DECODE_MAP = {
    0: 'A',
    1: 'B', 
    2: 'C',
    3: 'D'
}

# Reverse mapping for probabilities
GRADE_ENCODE_MAP = {v: k for k, v in GRADE_DECODE_MAP.items()}


def load_trained_model():
    """Load the trained model at startup."""
    global model, model_metadata
    
    # Try improved model first, then fall back to original
    model_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_models', 'best_model_improved.pkl'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_models', 'best_model.pkl')
    ]
    
    for model_path in model_paths:
        try:
            if os.path.exists(model_path):
                model, model_metadata = load_model(model_path)
                logger.info(f"Model loaded successfully from {model_path}")
                logger.info(f"Model metadata: {model_metadata}")
                return
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            continue
    
    logger.warning("No model file found. Please run main.py or main_improved.py first to train and save the model")


@app.on_event("startup")
async def startup_event():
    """Load model when the API starts."""
    load_trained_model()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "Student Grade Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_grade(features: StudentFeatures):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please ensure the model is trained and saved."
        )
    
    try:
        # Convert enums to encoded integers using mapping
        input_data = {
            'Gender': ENCODING_MAPS['Gender'][features.gender.value],
            'Transportation': ENCODING_MAPS['Transportation'][features.transportation.value],
            'Accommodation': ENCODING_MAPS['Accommodation'][features.accommodation.value],
            'MidExam': ENCODING_MAPS['MidExam'][features.mid_exam.value],
            'TakingNotes': ENCODING_MAPS['TakingNotes'][features.taking_notes.value]
        }
        
        logger.info(f"Encoded input for prediction: {input_data}")
        
        # Predict (this returns encoded integer)
        predicted_grade_encoded = model.predict_single(input_data)
        logger.info(f"Raw prediction (encoded): {predicted_grade_encoded}")
        
        # CRITICAL FIX: Decode the prediction back to letter grade
        predicted_grade = GRADE_DECODE_MAP[predicted_grade_encoded]
        logger.info(f"Decoded prediction: {predicted_grade}")
        
        # Get probabilities
        input_df = pd.DataFrame([input_data])
        probabilities_df = model.predict_proba(input_df)
        probabilities_encoded = probabilities_df.iloc[0].to_dict()
        
        # CRITICAL FIX: Decode probability keys from integers to letter grades
        probabilities = {}
        for encoded_grade, prob in probabilities_encoded.items():
            letter_grade = GRADE_DECODE_MAP[encoded_grade]
            probabilities[letter_grade] = prob
        
        logger.info(f"Decoded probabilities: {probabilities}")
        
        confidence = probabilities.get(predicted_grade, 0.0)
        
        model_info = {
            "version": model_metadata.get("version", "unknown") if model_metadata else "unknown",
            "k_value": model_metadata.get("k_value", "unknown") if model_metadata else "unknown",
            "test_accuracy": model_metadata.get("test_accuracy", "unknown") if model_metadata else "unknown"
        }
        
        return PredictionResponse(
            predicted_grade=predicted_grade,
            confidence=round(confidence, 4),
            probabilities={k: round(v, 4) for k, v in probabilities.items()},
            model_info=model_info
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(features_list: List[StudentFeatures]):
    """
    Predict grades for multiple students at once.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please ensure the model is trained and saved."
        )

    if not features_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty features list provided"
        )

    try:
        predictions = []

        for i, features in enumerate(features_list):
            input_data = {
                'Gender': ENCODING_MAPS['Gender'][features.gender.value],
                'Transportation': ENCODING_MAPS['Transportation'][features.transportation.value],
                'Accommodation': ENCODING_MAPS['Accommodation'][features.accommodation.value],
                'MidExam': ENCODING_MAPS['MidExam'][features.mid_exam.value],
                'TakingNotes': ENCODING_MAPS['TakingNotes'][features.taking_notes.value]
            }

            # Get encoded prediction and decode it
            predicted_grade_encoded = model.predict_single(input_data)
            predicted_grade = GRADE_DECODE_MAP[predicted_grade_encoded]
            
            input_df = pd.DataFrame([input_data])
            probabilities_df = model.predict_proba(input_df)
            probabilities_encoded = probabilities_df.iloc[0].to_dict()
            
            # Decode probabilities
            probabilities = {}
            for encoded_grade, prob in probabilities_encoded.items():
                letter_grade = GRADE_DECODE_MAP[encoded_grade]
                probabilities[letter_grade] = prob
            
            confidence = probabilities.get(predicted_grade, 0.0)

            predictions.append({
                "student_index": i,
                "predicted_grade": predicted_grade,
                "confidence": round(confidence, 4),
                "probabilities": {k: round(v, 4) for k, v in probabilities.items()}
            })

        logger.info(f"Batch prediction completed for {len(features_list)} students")
        return {"predictions": predictions}

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info")
async def get_model_info():
    """
    Get information about the loaded model.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        model_info = model.get_model_info()

        return {
            "model_info": model_info,
            "metadata": model_metadata or {},
            "status": "loaded",
            "grade_mapping": GRADE_DECODE_MAP
        }

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=f"Status code: {exc.status_code}"
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unexpected errors."""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )