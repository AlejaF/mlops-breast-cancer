from fastapi import FastAPI
import os

from app.schemas import (
    PredictionRequest,
    PredictionResponse
)

from app.model_loader import (
    load_model
)

from app.inference import (
    predict
)

from app.s3_utils import (
    append_prediction_to_s3
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "breast_cancer_v1.onnx"
)

ENVIRONMENT = os.getenv(
    "ENVIRONMENT",
    "dev"
)

session = load_model(
    MODEL_NAME
)


app = FastAPI(
    title="Breast Cancer MLOps Platform",
    description="""
    Prueba de API para inferencia de cáncer de mama utilizando modelos ONNX.

    Ambientes soportados:
    - DEV
    - PROD

    Características:
    - FastAPI
    - ONNX Runtime
    - Docker
    - CI/CD con GitHub Actions
    - Deploy automatizado
    """,
    version="1.0.0"
)


@app.get("/")
def root():

    return {
        "service": "Breast Cancer Prediction API",
        "model": MODEL_NAME
    }


@app.get("/sample-input")
def sample_input():

    return {
        "features": [
            17.99,
            10.38,
            122.8,
            1001.0,
            0.1184,
            0.2776,
            0.3001,
            0.1471,
            0.2419,
            0.0787,
            1.095,
            0.9053,
            8.589,
            153.4,
            0.0064,
            0.049,
            0.0537,
            0.0159,
            0.03,
            0.0062,
            25.38,
            17.33,
            184.6,
            2019.0,
            0.1622,
            0.6656,
            0.7119,
            0.2654,
            0.4601,
            0.1189
        ]
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def make_prediction(
    request: PredictionRequest
):

    result = predict(
        session,
        request.features
    )

    append_prediction_to_s3(
        environment=ENVIRONMENT,
        prediction=result["prediction"],
        label=result["label"]
    )

    return result
