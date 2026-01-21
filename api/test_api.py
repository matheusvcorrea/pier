# Lottery Accumulation Prediction API - Test Suite
# Comprehensive tests for the FastAPI application and model

import pytest
from fastapi.testclient import TestClient
from api.api import app
from api.schemas import LotteryDrawInput

# Create test client
client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Lottery Accumulation Prediction API"
    assert response.json()["status"] == "running"


def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "timestamp" in response.json()


def test_model_loading():
    """Test that the model loads correctly"""
    # This test verifies that the model can be loaded
    # The app should have loaded it during initialization
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] == True


def test_model_info_endpoint():
    """Test the model info endpoint"""
    response = client.get("/model-info")
    
    assert response.status_code == 200

    data = response.json()
    expected_model_type = "LGBMClassifier"

    assert data["model_type"] == expected_model_type
    assert data["version"] == "1.0"
    assert len(data["features"]) > 0


def test_prediction_endpoint_valid_input():
    """Test prediction endpoint with valid input"""
    # Create valid test input
    test_input = {
        "valorAcumuladoProximoConcurso": 1000000.0,
        "valorEstimadoProximoConcurso": 1500000.0,
        "valorArrecadado": 500000.0,
        "valorAcumuladoConcurso_0_5": 100000.0,
        "valorAcumuladoConcursoEspecial": 200000.0
    }
    
    response = client.post("/predict", json=test_input)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check that all required fields are present
    assert "prediction" in data
    assert "probability" in data
    assert "confidence" in data
    assert "model_version" in data
    assert "timestamp" in data
    
    # Check that prediction values are valid
    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability"] <= 1
    assert 0 <= data["confidence"] <= 1
    assert data["model_version"] in ["1.0", "latest"]


def test_prediction_endpoint_missing_field():
    """Test prediction endpoint with missing required field"""
    # Create input with missing field
    test_input = {
        "valorAcumuladoProximoConcurso": 1000000.0,
        "valorEstimadoProximoConcurso": 1500000.0,
        "valorArrecadado": 500000.0,
        # Missing 'valorAcumuladoConcurso_0_5' field
        "valorAcumuladoConcursoEspecial": 200000.0
    }
    
    response = client.post("/predict", json=test_input)
    
    # Should return 422 Unprocessable Entity for validation error
    assert response.status_code == 422
    assert "field required" in response.text.lower()


def test_prediction_endpoint_invalid_type():
    """Test prediction endpoint with invalid data type"""
    # Create input with invalid type
    test_input = {
        "valorAcumuladoProximoConcurso": "not_a_number",  # Should be float
        "valorEstimadoProximoConcurso": 1500000.0,
        "valorArrecadado": 500000.0,
        "valorAcumuladoConcurso_0_5": 100000.0,
        "valorAcumuladoConcursoEspecial": 200000.0
    }
    
    response = client.post("/predict", json=test_input)
    
    # Should return 422 Unprocessable Entity for validation error
    assert response.status_code == 422
    assert "validation error" in response.text.lower() or "input should be a valid" in response.text.lower()


def test_input_validation():
    """Test input validation with Pydantic model"""
    # Test valid input
    valid_input = LotteryDrawInput(
        valorAcumuladoProximoConcurso=1000000.0,
        valorEstimadoProximoConcurso=1500000.0,
        valorArrecadado=500000.0,
        valorAcumuladoConcurso_0_5=100000.0,
        valorAcumuladoConcursoEspecial=200000.0
    )
    
    # Should not raise any validation errors
    assert valid_input.valorAcumuladoProximoConcurso == 1000000.0
    assert valid_input.valorEstimadoProximoConcurso == 1500000.0
    
    # Test invalid input
    with pytest.raises(Exception):
        # Should raise validation error
        LotteryDrawInput(
            valorAcumuladoProximoConcurso="invalid",  # Should be float
            valorEstimadoProximoConcurso=1500000.0,
            valorArrecadado=500000.0,
            valorAcumuladoConcurso_0_5=100000.0,
            valorAcumuladoConcursoEspecial=200000.0
        )