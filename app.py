"""
app.py
API REST con Flask para la predicción de enfermedad pulmonar
usando el modelo Random Forest entrenado.

Autor: Grupo - Curso de Programación (6to ciclo - Ing. de Sistemas)

Endpoints:
  GET  /              → Sirve el formulario HTML
  POST /predict       → Recibe JSON, retorna predicción
  POST /send-report   → Recibe PDF en base64 y lo envía por correo
  GET  /health        → Estado de la API
"""

import pickle
import os
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_mail import Mail, Message

# ── Inicialización ────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Configuración de correo (Gmail) ───────────────────────────────────────────
# Las credenciales se leen desde variables de entorno de Render.
# No pongas tu contraseña directamente aquí.
app.config['MAIL_SERVER']         = 'smtp.gmail.com'
app.config['MAIL_PORT']           = 587
app.config['MAIL_USE_TLS']        = True
app.config['MAIL_USERNAME']       = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD']       = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_USERNAME')

mail = Mail(app)

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
        ...
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

    prediccion     = int(MODEL.predict(entrada)[0])
    probabilidades = MODEL.predict_proba(entrada)[0]

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


@app.route('/send-report', methods=['POST'])
def send_report():
    """
    Recibe el reporte PDF en base64 y lo envía al correo del paciente.

    Body esperado (JSON):
    {
        "email":        "paciente@ejemplo.com",
        "patient_name": "Juan García",
        "pdf_base64":   "<string base64 del PDF>"
    }
    """
    # Verificar que las credenciales de correo estén configuradas
    if not os.environ.get('MAIL_USERNAME') or not os.environ.get('MAIL_PASSWORD'):
        return jsonify({
            'error': 'Servicio de correo no configurado en el servidor.'
        }), 503

    data         = request.get_json(silent=True)
    to_email     = (data.get('email')        or '').strip()
    patient_name = (data.get('patient_name') or 'Paciente').strip()
    pdf_b64      = (data.get('pdf_base64')   or '').strip()

    if not to_email or not pdf_b64:
        return jsonify({'error': 'Faltan datos: email o pdf_base64.'}), 400

    # Decodificar el PDF de base64 a bytes
    try:
        pdf_bytes = base64.b64decode(pdf_b64)
    except Exception:
        return jsonify({'error': 'El PDF recibido no es válido.'}), 400

    # Nombre del archivo adjunto
    nombre_archivo = f"PulmoCheck_Reporte_{patient_name.replace(' ', '_')}.pdf"

    try:
        msg = Message(
            subject=f'PulmoCheck — Reporte de Análisis Pulmonar',
            recipients=[to_email]
        )
        msg.body = (
            f'Estimado/a {patient_name},\n\n'
            'Adjunto encontrará su reporte de análisis pulmonar generado por PulmoCheck.\n\n'
            'Este reporte ha sido generado mediante un modelo de Machine Learning '
            '(Random Forest) y tiene carácter informativo. No reemplaza el diagnóstico '
            'de un médico especialista.\n\n'
            'Se recomienda compartir este reporte con su médico de cabecera.\n\n'
            '— PulmoCheck · Ing. de Sistemas — 6.° Ciclo'
        )
        msg.attach(nombre_archivo, 'application/pdf', pdf_bytes)
        mail.send(msg)
        return jsonify({'success': True, 'message': f'Reporte enviado a {to_email}'})

    except Exception as e:
        return jsonify({'error': f'Error al enviar el correo: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)