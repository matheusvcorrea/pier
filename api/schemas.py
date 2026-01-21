from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class LotteryDrawInput(BaseModel):
    valorAcumuladoProximoConcurso: float = Field(..., description="Accumulated value for next draw", examples=[1000000.0])
    valorEstimadoProximoConcurso: float = Field(..., description="Estimated value for next draw", examples=[1500000.0])
    valorArrecadado: float = Field(..., description="Amount collected", examples=[500000.0])
    valorAcumuladoConcurso_0_5: float = Field(..., description="Accumulated value for draws 0-5", examples=[100000.0])
    valorAcumuladoConcursoEspecial: float = Field(..., description="Accumulated value for special draws", examples=[200000.0])


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = No accumulation, 1 = Accumulation")
    probability: float = Field(..., description="Probability of accumulation (0-1)")
    confidence: float = Field(..., description="Confidence in prediction (0-1)")
    model_version: str = Field(..., description="Version of the model used")
    timestamp: str = Field(..., description="Timestamp of prediction")
    model_info: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Additional model information"
    )
