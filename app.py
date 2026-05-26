"""
app.py
API REST con Flask para la predicción de enfermedad pulmonar
usando el modelo Random Forest entrenado.

Autor: Grupo - Curso de Programación (6to ciclo - Ing. de Sistemas)

Encodings del dataset:
  AGE                    : numérico entero
  GENDER                 : Masculino=0, Femenino=1
  SMOKING                : No=0, Sí=1
  FINGER_DISCOLORATION   : No=0, Sí=1
  MENTAL_STRESS          : No=0, Sí=1
  EXPOSURE_TO_POLLUTION  : No=0, Sí=1
  LONG_TERM_ILLNESS      : No=0, Sí=1
  IMMUNE_WEAKNESS        : No=0, Sí=1
  BREATHING_ISSUE        : No=0, Sí=1
  ALCOHOL_CONSUMPTION    : No=0, Sí=1
  THROAT_DISCOMFORT      : No=0, Sí=1
  CHEST_TIGHTNESS        : No=0, Sí=1
  FAMILY_HISTORY         : No=0, Sí=1
  SMOKING_FAMILY_HISTORY : No=0, Sí=1
  STRESS_IMMUNE          : No=0, Sí=1
  ENERGY_LEVEL           : decimal (23.26 – 83.05)
  OXYGEN_SATURATION      : decimal (89.92 – 99.80)
  PULMONARY_DISEASE (y)  : No=0, Sí=1

Endpoints:
  GET  /          → Sirve el formulario HTML
  POST /predict   → Recibe JSON, retorna predicción
  GET  /health    → Estado de la API
"""

import pickle
import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ── Inicialización ────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Carga del modelo entrenado ────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)

    MODEL      = model_data['model']
    FEAT_NAMES = model_data['feature_names']
    print(f"✓ Modelo cargado | Features: {FEAT_NAMES}")

except FileNotFoundError:
    MODEL      = None
    FEAT_NAMES = []
    print("⚠ model.pkl no encontrado. Ejecuta train_model.py primero.")


# ── Rutas ─────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    """Sirve el formulario HTML principal."""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """Endpoint de comprobación del estado de la API."""
    return jsonify({
        'status':         'ok',
        'modelo_cargado': MODEL is not None,
        'features':       FEAT_NAMES
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Recibe los datos del formulario y retorna la predicción.

    Body esperado (JSON):
    {
        "AGE": 45,
        "GENDER": 1,
        "SMOKING": 0,
        "FINGER_DISCOLORATION": 1,
        "MENTAL_STRESS": 1,
        "EXPOSURE_TO_POLLUTION": 0,
        "LONG_TERM_ILLNESS": 1,
        "IMMUNE_WEAKNESS": 0,
        "BREATHING_ISSUE": 1,
        "ALCOHOL_CONSUMPTION": 0,
        "THROAT_DISCOMFORT": 1,
        "CHEST_TIGHTNESS": 1,
        "FAMILY_HISTORY": 0,
        "SMOKING_FAMILY_HISTORY": 1,
        "STRESS_IMMUNE": 0,
        "ENERGY_LEVEL": 45.5,
        "OXYGEN_SATURATION": 94.3
    }

    Respuesta (JSON):
    {
        "prediccion": 1,
        "etiqueta": "Sí",
        "probabilidad_no": 0.18,
        "probabilidad_si": 0.82,
        "riesgo": "Alto"
    }
    """
    if MODEL is None:
        return jsonify({
            'error': 'Modelo no disponible. Ejecuta train_model.py primero.'
        }), 503

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'error': 'Se esperaba un cuerpo JSON válido.'}), 400

    try:
        valores = [data[feat] for feat in FEAT_NAMES]
    except KeyError as e:
        return jsonify({'error': f'Campo faltante: {str(e)}'}), 400

    try:
        entrada = np.array([valores], dtype=float)
    except (ValueError, TypeError):
        return jsonify({'error': 'Todos los valores deben ser numéricos.'}), 400

    prediccion     = int(MODEL.predict(entrada)[0])       # 0 o 1
    probabilidades = MODEL.predict_proba(entrada)[0]      # [P(0), P(1)]

    prob_no  = float(probabilidades[0])
    prob_si  = float(probabilidades[1])
    etiqueta = "Sí" if prediccion == 1 else "No"

    if prob_si >= 0.75:
        riesgo = 'Alto'
    elif prob_si >= 0.50:
        riesgo = 'Moderado'
    else:
        riesgo = 'Bajo'

    return jsonify({
        'prediccion':      prediccion,
        'etiqueta':        etiqueta,
        'probabilidad_no': round(prob_no, 4),
        'probabilidad_si': round(prob_si, 4),
        'riesgo':          riesgo
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)