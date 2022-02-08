import json
import pickle
import numpy as np
import os

def init():
    global model, scaler
    version = max(os.listdir(os.path.join(os.getenv("AZUREML_MODEL_DIR"), "wine-quality-lr/")))
    file_name = f"wine-quality-lr/{version}/model.pkl"
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), file_name)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    version = max(os.listdir(os.path.join(os.getenv("AZUREML_MODEL_DIR"), "wine-quality-scaler/")))
    file_name = f"wine-quality-scaler/{version}/scaler.pkl"
    scaler_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), file_name)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    
def run(raw_data):
    data = np.array(json.loads(raw_data)["data"])
    prepped = scaler.transform(data)
    predictions = model.predict(prepped)
    return predictions.tolist()