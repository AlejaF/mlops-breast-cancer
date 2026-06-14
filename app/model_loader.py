from pathlib import Path

import onnxruntime as ort

from app.s3_utils import (
    download_model_from_s3
)


BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = BASE_DIR / "artifacts"


def load_model(model_name: str):

    model_path = (
        ARTIFACTS_DIR / model_name
    )

    if not model_path.exists():

        download_model_from_s3(
            model_name
        )

    session = ort.InferenceSession(
        str(model_path),
        providers=[
            "CPUExecutionProvider"
        ]
    )

    return session