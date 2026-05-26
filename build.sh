#!/bin/bash
# build.sh
# Render ejecuta este script antes de iniciar el servidor.
# Entrena el modelo y guarda model.pkl si no existe.

echo "── Entrenando modelo Random Forest ──"
python train_model.py
echo "── Modelo listo. Iniciando servidor ──"