"""
train_model.py
Entrena el modelo Random Forest con el dataset de enfermedad pulmonar
y lo guarda como modelo serializado (.pkl) para usarlo en la API.

Autor: Grupo - Curso de Programación (6to ciclo - Ing. de Sistemas)
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── 1. Cargar el dataset ──────────────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'pulmon1.csv')

dataframe = pd.read_csv(DATASET_PATH, sep=',')
clasificadores = ['No', 'Si']

print("✓ Dataset cargado correctamente")
print(f"  Filas: {dataframe.shape[0]}  |  Columnas: {dataframe.shape[1]}")
print(f"  Columnas: {list(dataframe.columns)}\n")

# ── 2. Separar features y variable objetivo ───────────────────────────────────
y = dataframe['PULMONARY_DISEASE']
x = dataframe.drop(['PULMONARY_DISEASE'], axis=1)

FEATURE_NAMES = list(x.columns)
print(f"✓ Features ({len(FEATURE_NAMES)}): {FEATURE_NAMES}\n")

# ── 3. División train/test (70% / 30%) ────────────────────────────────────────
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=30
)

# ── 4. Entrenamiento del modelo Random Forest final (n_estimators=150) ────────
rf_final = RandomForestClassifier(n_estimators=150, random_state=30)
rf_final.fit(x_train, y_train)
print("✓ Modelo entrenado exitosamente (n_estimators=150)")

# ── 5. Evaluación del modelo ──────────────────────────────────────────────────
predicciones = rf_final.predict(x_test)
accuracy     = accuracy_score(y_test, predicciones)

print(f"\n── Evaluación del modelo ──────────────────────────────────────")
print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\n{classification_report(y_test, predicciones, target_names=clasificadores)}")
print(f"  Matriz de confusión:\n{confusion_matrix(y_test, predicciones)}")

# ── 6. Guardar modelo y nombres de columnas ───────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

model_data = {
    'model':         rf_final,
    'feature_names': FEATURE_NAMES
}

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✓ Modelo guardado en: {MODEL_PATH}")
print("  Listo para usar en la API Flask.")