from flask import Flask, jsonify, request, Blueprint, make_response
from PathModel import PathModel
import random 
import pandas as pd
from services import initializeService

route_bp = Blueprint("test", __name__)

@route_bp.route('/test')
def testing():
    return "Hello world"

#Metodo inicial para el usuario
@route_bp.route('/init')
def init():
    #Este debe ser el uuid del nuestro modelo
    numero = random.random()
    header, matriz_values = initializeService.readCSV("datasets/example.csv")

    resp = jsonify({"message":"cookie created", "headerTest": header, "matriz_values:": matriz_values})
    resp.set_cookie("current_model",str(numero))
    return resp

# post en el diccionario global
@route_bp.route('/global/<clave>/<valor>', methods = ['POST'])
def diccionario(clave,valor):
    PathModel[clave] = valor
    return jsonify({'message': 'agregado','diccionario':PathModel})

#Get diccionario
@route_bp.route('/global')
def get_diccionario():
    return jsonify(PathModel)

@route_bp.route('/train', methods = ['POST'])
def train():
    return 'trained'

@route_bp.route('/predict', methods = ['POST'])
def predict():
    return 'predicted'

#Exportar el modelo
@route_bp.route('/export/<string:uuid>', methods = ['GET'])
def export_model(uuid):
    path = PathModel[uuid]
    csv = pd.read_csv(path)
    return csv.to_json(orient='records')
