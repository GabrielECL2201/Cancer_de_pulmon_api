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
import smtplib
import numpy as np
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from email.mime.application import MIMEApplication
from flask      import Flask, request, jsonify, render_template
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
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':         'ok',
        'modelo_cargado': MODEL is not None,
        'features':       FEAT_NAMES
    })


@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({'error': 'Modelo no disponible.'}), 503

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
    Recibe el PDF en base64 y lo envía al correo del paciente
    usando smtplib con Gmail (sin dependencias externas).

    Variables de entorno requeridas en Render:
      MAIL_USERNAME → correo Gmail remitente
      MAIL_PASSWORD → App Password de Google (16 caracteres)
    """
    mail_user = os.environ.get('MAIL_USERNAME', '').strip()
    mail_pass = os.environ.get('MAIL_PASSWORD', '').strip()

    if not mail_user or not mail_pass:
        return jsonify({'error': 'Servicio de correo no configurado en el servidor.'}), 503

    data         = request.get_json(silent=True)
    to_email     = (data.get('email')        or '').strip()
    patient_name = (data.get('patient_name') or 'Paciente').strip()
    pdf_b64      = (data.get('pdf_base64')   or '').strip()

    if not to_email or not pdf_b64:
        return jsonify({'error': 'Faltan datos: email o pdf_base64.'}), 400

    try:
        pdf_bytes = base64.b64decode(pdf_b64)
    except Exception:
        return jsonify({'error': 'El PDF recibido no es válido.'}), 400

    nombre_archivo = f"PulmoCheck_Reporte_{patient_name.replace(' ', '_')}.pdf"

    try:
        # Construir el correo
        msg = MIMEMultipart()
        msg['From']    = mail_user
        msg['To']      = to_email
        msg['Subject'] = 'PulmoCheck — Reporte de Analisis Pulmonar'

        cuerpo = (
            f'Estimado/a {patient_name},\n\n'
            'Adjunto encontrara su reporte de analisis pulmonar generado por PulmoCheck.\n\n'
            'Este reporte ha sido generado mediante un modelo de Machine Learning '
            '(Random Forest) y tiene caracter informativo. No reemplaza el diagnostico '
            'de un medico especialista.\n\n'
            'Se recomienda compartir este reporte con su medico de cabecera.\n\n'
            '— PulmoCheck · Ing. de Sistemas — 6. Ciclo'
        )
        msg.attach(MIMEText(cuerpo, 'plain'))

        # Adjuntar el PDF
        adjunto = MIMEApplication(pdf_bytes, _subtype='pdf')
        adjunto.add_header('Content-Disposition', 'attachment', filename=nombre_archivo)
        msg.attach(adjunto)

        # Enviar via Gmail SMTP
        with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(mail_user, mail_pass)
            smtp.send_message(msg)

        return jsonify({'success': True, 'message': f'Reporte enviado a {to_email}'})

    except smtplib.SMTPAuthenticationError:
        return jsonify({'error': 'Credenciales de correo incorrectas. Verifica MAIL_USERNAME y MAIL_PASSWORD en Render.'}), 500
    except smtplib.SMTPException as e:
        return jsonify({'error': f'Error SMTP: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Error inesperado: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)