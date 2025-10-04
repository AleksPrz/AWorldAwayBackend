from flask import Flask, jsonify, request, Blueprint, make_response
from PathModel import PathModel
import random 


route_bp = Blueprint("test", __name__)

@route_bp.route('/test')
def testing():
    return "Hello world"

@route_bp.route('/init')
def init():
    numero = random.random()
    resp = jsonify({"message":"cookie created"})
    resp.set_cookie("current_model",str(numero))
    return resp

@route_bp.route('/global/<clave>/<valor>')
def diccionario(clave,valor):
    PathModel[clave] = valor
    return jsonify({'message': 'agregado','diccionario':PathModel})

@route_bp.route('/global')
def get_diccionario():
    return jsonify(PathModel)