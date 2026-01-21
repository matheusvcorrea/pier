import logging
import os
import joblib

logger = logging.getLogger(__name__)

# Default model paths
MODEL_NAME = "lottery_accumulation_model"
MODEL_VERSION = "latest"
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "lottery_model.pkl")
MLFLOW_MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"


def load_model(path: str = "lottery_model.pkl"):
    """
    Load model from MLflow first, fall back to local joblib file.
    
    Returns:
        Tuple of (loaded model, model_version)
    """
    model_version = MODEL_VERSION

    try:
        import mlflow
        import mlflow.lightgbm
        
        logger.info(f"Attempting to load model from MLflow: {MLFLOW_MODEL_URI}")
        client = mlflow.tracking.MlflowClient()

        # Load the mlflow model based on the URI
        model = mlflow.lightgbm.load_model(MLFLOW_MODEL_URI)
        model_versions = client.search_model_versions(f"name='lottery_accumulation_model'")
        # version_details = client.get_model_version(name=model_name, version=model_version)
        
        if model_versions:
            model_version = model_versions[0].version
            logger.info(f"Model loaded successfully from MLflow with version: {model_version}")
        else:
            logger.warning("Could not determine model version from MLflow, using 'latest'")
            model_version = "latest"
        
        return model, model_version
        
    except Exception as e:
        logger.warning(f"Failed to load model from MLflow: {e}")
    
    # Fall back to local joblib file
    try:
        model = joblib.load(LOCAL_MODEL_PATH)
        logger.info("Model loaded successfully from local file")
        return model, model_version
    except Exception as e:
        logger.error(f"Failed to load model from local file: {e}")
        raise