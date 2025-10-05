import base64
import pandas as pd
from PathModels import PathModels
import json
import uuid

def defaults_values(path):
    csv = pd.read_csv(path, nrows = 10)
    header = list(csv.columns)
    values_matriz = csv.values.tolist()

    with open('models/KOI_best_params.json', 'r') as f:
        parameters_default = json.load(f)
    
    return header, values_matriz, parameters_default

def generateUUID():
    uuid_new = uuid.uuid4()
    return uuid_new

def create_new_path(uuid):
    path = 'models/'+uuid+'.pkl'
    return path

def create_new_path_scaler(uuid):
    path = 'models/'+uuid+'_scaler.pkl'
    return path


def image_to_base64(path):
    # Abre la imagen en modo binario
    with open(path, "rb") as image_file:
        # Lee los bytes
        image_bytes = image_file.read()
        # Convierte los bytes a base64
        base64_bytes = base64.b64encode(image_bytes)
        # Convierte de bytes a string (UTF-8)
        base64_string = base64_bytes.decode("utf-8")
        return base64_string