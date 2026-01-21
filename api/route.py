"""
Route definitions for the Lottery Accumulation Prediction API
"""
from fastapi import APIRouter, HTTPException
from api.estimator import load_model
from api.schemas import LotteryDrawInput, PredictionResponse
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

model_type = "LGBMClassifier" # used for model details att response

# Load estimator model
model, model_version = load_model()

@router.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "Lottery Accumulation Prediction API",
        "version": "1.0.0",
        "status": "running"
    }


@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": True,
        "model_version": model_version
    }


@router.post("/predict", response_model=PredictionResponse)
def predict_accumulation(input_data: LotteryDrawInput):
    """
    Predict whether the next lottery draw will accumulate
    
    Args:
        input_data: Lottery draw information
        
    Returns:
        Prediction with probability and confidence
    """
    try:
        input_df = pd.DataFrame([input_data.model_dump()])
        
        if input_df.isnull().any().any():
            raise HTTPException(
                status_code=400,
                detail="Input data contains null values"
            )
        
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        
        # Prepare response
        response = {
            "prediction": int(prediction[0]),
            "probability": float(probability[0][1]),  # Probability of accumulation
            "confidence": float(max(probability[0])),
            "model_version": model_version,
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "model_type": model_type,
                "features_used": list(input_data.model_dump().keys())
            }
        }
        
        logger.info(f"Prediction made: {response['prediction']}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/model-info")
def get_model_info():
    """Get information about the loaded model"""
    try:
        # Get model information from the pipeline
        model_info = {
            "model_type": model_type,
            "version": "1.0",
            "features": [
                "valorAcumuladoProximoConcurso",
                "valorEstimadoProximoConcurso",
                "valorArrecadado",
                "valorAcumuladoConcurso_0_5",
                "valorAcumuladoConcursoEspecial"
            ],
            "target": "acumulou (0=No, 1=Yes)",
            "training_date": datetime.now().strftime("%Y-%m-%d")
        }
        
        return model_info
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )
