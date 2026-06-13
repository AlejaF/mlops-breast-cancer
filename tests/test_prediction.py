from app.model_loader import load_model
from app.inference import predict


def test_prediction():

    session = load_model(
        "breast_cancer_v1.onnx"
    )

    features = [
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

    result = predict(
        session,
        features
    )

    assert result["prediction"] in [0, 1]

    assert result["label"] in [
        "malignant",
        "benign"
    ]