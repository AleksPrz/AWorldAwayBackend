from flask import Flask, jsonify, request, Blueprint, make_response, send_file
from PathModels import PathModels
import random 
import pandas as pd
from services import initializeService
import os


route_bp = Blueprint("routes", __name__)

#Metodo inicial para el usuario
@route_bp.route('/init')
def init():
    #Este debe ser el uuid del nuestro modelo
    header, matriz_values, parameters_default = initializeService.defaults_values("datasets/koi_processed.csv")
    resp = jsonify({"headerTest": header, "matriz_values:": matriz_values,"parameters_default":parameters_default, "message":"cookie created"})
    resp.set_cookie("current_model","current_model")
    return resp

#reset model
@route_bp.route('/reset/values')
def reset_values():
    PathModels.clear()
    PathModels['current_model'] = 'models/KOI.pkl'
    return jsonify({'message': 'reset complete'})

#Exportar el modelo funcional
@route_bp.route('/export/<string:uuid>', methods = ['GET'])
def export_model(uuid):
    path = PathModels[uuid]
    print(path)
    if path is None:
        return jsonify({'message':"Archivo no encontrado"})
    filename = os.path.basename(path)
    return send_file(
        path,
        as_attachment = True,
        download_name = filename,
        mimetype = "aplication/octet-stream"
    )   

@route_bp.route('/train/gbt', methods = ['POST'])
def train(learning_rate,max_depth,min_samples_split,n_estimators,subsample):
    #algoritmo, parametros,
    #LLamada para training
    uuid = initializeService.generateUUID()
    path = initializeService.create_new_path(uuid)
    resp = jsonify({'message': 'model trained'})
    resp.set_cookie('current_model',uuid)
    return resp

@route_bp.route('/predict', methods = ['POST'])
def predict():
    return 'predicted'