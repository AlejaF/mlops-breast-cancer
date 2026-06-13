from pathlib import Path

import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


# ==========================================================
# Crear carpeta de artefactos
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ==========================================================
# Cargar dataset
# ==========================================================

data = load_breast_cancer()

X = data.data
y = data.target

feature_names = data.feature_names


# ==========================================================
# Train/Test split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================================
# Guardar dataset de prueba
# ==========================================================

test_df = pd.DataFrame(
    X_test,
    columns=feature_names
)

test_df["target"] = y_test

test_path = ARTIFACTS_DIR / "breast_cancer_test.csv"

test_df.to_csv(
    test_path,
    index=False
)

print(f"Dataset de prueba guardado en: {test_path}")


# ==========================================================
# Modelo v1
# ==========================================================

model_v1 = RandomForestClassifier(
    n_estimators=10,
    max_depth=3,
    random_state=42
)

model_v1.fit(   
    X_train,
    y_train
)

pred_v1 = model_v1.predict(X_test)

acc_v1 = accuracy_score(
    y_test,
    pred_v1
)

print(f"Accuracy v1: {acc_v1:.4f}")


# ==========================================================
# Exportar v1 a ONNX
# ==========================================================

initial_type = [
    ("float_input", FloatTensorType([None, X.shape[1]]))
]

onnx_v1 = convert_sklearn(
    model_v1,
    initial_types=initial_type
)

v1_path = ARTIFACTS_DIR / "breast_cancer_v1.onnx"

with open(v1_path, "wb") as f:
    f.write(onnx_v1.SerializeToString())

print(f"Modelo v1 guardado en: {v1_path}")


# ==========================================================
# Modelo v2
# ==========================================================

model_v2 = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model_v2.fit(
    X_train,
    y_train
)

pred_v2 = model_v2.predict(X_test)

acc_v2 = accuracy_score(
    y_test,
    pred_v2
)

print(f"Accuracy v2: {acc_v2:.4f}")


# ==========================================================
# Exportar v2 a ONNX
# ==========================================================

onnx_v2 = convert_sklearn(
    model_v2,
    initial_types=initial_type
)

v2_path = ARTIFACTS_DIR / "breast_cancer_v2.onnx"

with open(v2_path, "wb") as f:
    f.write(onnx_v2.SerializeToString())

print(f"Modelo v2 guardado en: {v2_path}")


# ==========================================================
# Resumen
# ==========================================================

print("\nProceso finalizado correctamente.")
print(f"v1 accuracy: {acc_v1:.4f}")
print(f"v2 accuracy: {acc_v2:.4f}")