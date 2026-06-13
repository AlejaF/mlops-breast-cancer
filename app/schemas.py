from pydantic import BaseModel, Field
from typing import List


class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=30,
        max_length=30
    )


class PredictionResponse(BaseModel):
    prediction: int
    label: str