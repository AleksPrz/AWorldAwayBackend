import pandas as pd

def readCSV(path):
    csv = pd.read_csv(path, nrows = 10)

    header = list(csv.columns)

    values_matriz = csv.values.tolist()

    return header, values_matriz