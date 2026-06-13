from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

from app.model_loader import load_model
from app.inference import predict


def test_accuracy():

    test_data_path = Path(
        "artifacts/breast_cancer_test.csv"
    )

    df = pd.read_csv(
        test_data_path
    )

    X = df.drop(
        columns=["target"]
    )

    y_true = df["target"]

    session = load_model(
        "breast_cancer_v1.onnx"
    )

    predictions = []

    for _, row in X.iterrows():

        result = predict(
            session,
            row.tolist()
        )

        predictions.append(
            result["prediction"]
        )

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    assert accuracy >= 0.90