from pathlib import Path

import onnxruntime as ort


BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = BASE_DIR / "artifacts"


def load_model(model_name: str):
    """
    Carga un modelo ONNX y retorna una sesión ONNX Runtime.
    """

    model_path = ARTIFACTS_DIR / model_name

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {model_path}"
        )

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"]
    )

    return session