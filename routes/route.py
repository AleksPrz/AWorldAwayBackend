from flask import Flask, jsonify, request, Blueprint, make_response, send_file
from PathModels import PathModels
import joblib
import random 
import pandas as pd
from services import initializeService
from ML import ml
import os


route_bp = Blueprint("routes", __name__)

#Metodo inicial para el usuario
@route_bp.route('/init')
def init():
    #Este debe ser el uuid del nuestro modelo
    header, matriz_values, parameters_default = initializeService.defaults_values("datasets/koi_processed.csv")
    resp = jsonify({"headerTest": header, "matriz_values:": matriz_values,"parameters_default":parameters_default, "message":"cookie created"})
    resp.set_cookie("current_model","KOI")
    return resp

#reset model
@route_bp.route('/reset/values')
def reset_values():
    PathModels.clear()
    PathModels['current_model'] = ('models/KOI.pkl','models/KOI_scaler.kpl')
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

@route_bp.route('/train/gbt/koi', methods = ['POST'])
def train():
    #algoritmo, parametros,
    data = request.get_json()
    df = pd.read_csv('datasets/koi_processed')
    model, scaler, graphics = ml.train_gbt(df,'koi_disposition',data['n_estimators'], data['learning_rate'], data['max_depth'],
                                            data['min_samples_split'], data['train_size'], data.get('scaler_type'))
    
    uuid = initializeService.generateUUID()
    model_path = initializeService.create_new_path(uuid)
    scaler_path = initializeService.create_new_path_scaler(uuid)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    PathModels[uuid] = (model_path, scaler_path)
    
    resp = jsonify(graphics)
    resp.set_cookie('current_model',uuid)
    return resp

@route_bp.route('/train/gbt/csv', mrthods = ["POST"])
def trainCSV():
    archivo = request.files.get('file')  # el nombre del input type=file
    if archivo:
        # Convertir CSV a DataFrame de pandas
        df = pd.read_csv(archivo)
        
        data = request.get_json()
        model,scaler,graphics = ml.train_gbt(df,data['target_column'],data['n_estimators'], data['learning_rate'], data['max_depth'],
                                            data['min_samples_split'], data['train_size'], 'standard')
        
        uuid = initializeService.generateUUID()
        model_path = initializeService.create_new_path(uuid)
        scaler_path = initializeService.create_new_path_scaler(uuid)

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        PathModels[uuid] = (model_path, scaler_path)
        
        resp = jsonify(graphics)
        resp.set_cookie('current_model',uuid)
        return resp
    return jsonify({'message': 'File error'})




@route_bp.route('/predict', methods = ['POST'])
def predict():
    #model_path = PathModels[]
    return 'predicted'

@route_bp.route('/predict/batch', methods = ['POST'])
def predict():
    model_id = request.cookies.get('current_model')
    model_path, scaler_path = PathModels[model_id]
    data = request.get_json()
    #model_path = PathModels[]
    return 'predicted'