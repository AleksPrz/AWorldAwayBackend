from flask import Flask, jsonify, request, Blueprint, make_response, send_file
from PathModels import PathModels
import random 
import pandas as pd
from services import initializeService

route_bp = Blueprint("routes", __name__)

#Metodo inicial para el usuario
@route_bp.route('/init')
def init():
    #Este debe ser el uuid del nuestro modelo
    header, matriz_values = initializeService.readCSV("datasets/example.csv")
    resp = jsonify({"message":"cookie created", "headerTest": header, "matriz_values:": matriz_values})
    resp.set_cookie("current_model","current_model")
    return resp


# post en el diccionario global
@route_bp.route('/global/<string:value>', methods = ['POST'])
def diccionario():
    PathModels['current_model'] = request.json['value']
    return jsonify({'message': 'agregado','diccionario':PathModels})

#reset model
@route_bp.route('/reset/values')
def reset_values():
    PathModels.clear()
    #Cambiar el valor para que apunte al path de KOI_MODEL
    PathModels['KOI_Model'] = "Reseteaste el diccionario"
    return jsonify({'message': 'reset complete'})

#Exportar el modelo HAY QUE CHECARLO
@route_bp.route('/export/<string:uuid>', methods = ['GET'])
def export_model(uuid):
    path = PathModels[uuid]

    if path is None:
        return jsonify({'message':"Archivo no encontrado"})
    return send_file(
        path,
        as_attachment = True,
        download_name = uuid + "Model",
        mimetype = "aplication/octet-stream"
    )   
    #csv = pd.read_csv(path)
    #return csv.to_json(orient='records')

@route_bp.route('/train', methods = ['POST'])
def train():
    #algoritmo, parametros,
    resp = jsonify({})
    resp.set_cookie('current_model',"ultimo modelo")
    return 'trained'

@route_bp.route('/predict', methods = ['POST'])
def predict():
    return 'predicted'