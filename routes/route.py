from flask import Flask, jsonify, request, Blueprint

route_bp = Blueprint("test", __name__)

@route_bp.route('/test')
def testing():
    return "Hello world"