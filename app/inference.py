import numpy as np


def predict(session, features):
    """
    Ejecuta una predicción usando una sesión ONNX Runtime.
    """

    input_name = session.get_inputs()[0].name

    input_data = np.array(
        [features],
        dtype=np.float32
    )

    outputs = session.run(
        None,
        {
            input_name: input_data
        }
    )

    prediction = int(outputs[0][0])

    label = (
        "malignant"
        if prediction == 0
        else "benign"
    )

    return {
        "prediction": prediction,
        "label": label
    }