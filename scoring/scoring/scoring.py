
import json
import pickle
import numpy as np
import os

def init():
    global model, scaler
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "wine-quality-lr/1/model.pkl")
    print(model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    scaler_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "wine-quality-scaler/1/scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    
def run(raw_data):
    data = np.array(json.loads(raw_data)["data"])
    prepped = scaler.transform(data)
    predictions = model.predict(prepped)
    return predictions.tolist()

 

