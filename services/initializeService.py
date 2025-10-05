import pandas as pd
from PathModels import PathModels
import json

def defaults_values(path):
    csv = pd.read_csv(path, nrows = 10)
    header = list(csv.columns)
    values_matriz = csv.values.tolist()

    with open('models/KOI_best_params.json', 'r') as f:
        parameters_default = json.load(f)
    
    return header, values_matriz, parameters_default

def generateUUID():
    uuid = uuid.uuid4()
    return uuid

def create_new_path(uuid):
    PathModels[uuid] = 'models/'+uuid+'.pkl'
    return 'path created'