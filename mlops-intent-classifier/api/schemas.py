from pydantic import BaseModel, Field
from typing import List, Optional


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, example="What's the weather like today?")


class IntentPrediction(BaseModel):
    intent: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    text: str
    prediction: IntentPrediction
    top_k: List[IntentPrediction]
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_stage: str


class VersionResponse(BaseModel):
    model_name: str
    model_version: str
    model_stage: str
    run_id: Optional[str] = None
