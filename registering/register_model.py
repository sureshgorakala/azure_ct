from azureml.core import Run, Model
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", dest="model_name", type=str)
args = parser.parse_args()

run = Run.get_context()
workspace = run.experiment.workspace

def register_models():
    register = True
    # get new model metrics
    new_model_metrics = run.parent.get_metrics()
    new_test_accuracy = new_model_metrics["test_metrics"]["accuracy"]
    
    model_preexists = [model for model in Model.list(ws) if model.name==args.model_name]
    retraining = True if model_preexists else False
    
    if retraining:
        # get old model metrics
        old_model = Model(workspace=ws, name=args.model_name)
        old_model_metrics = old_model.run.parent.get_metrics()
        old_test_accuracy = old_model_metrics["test_metrics"]["accurracy"]
        
        if new_test_accuracy <= old_model_accuracy:
            register=False
    
    if register:
        model_path = os.path.join("outputs", "model.pkl")
        model = run.parent.register_model(model_name="wine-quality-lr", 
               model_path=model_path,
               description="lr model for wine quality",
               tags = {"dataset": "wine_train"}
              )
        model_path = os.path.join("outputs", "scaler.pkl")
        model = run.parent.register_model(model_name="wine-quality-scaler", 
               model_path=model_path,
               description="scaler model for wine quality",
               tags = {"dataset": "wine_train"}
              )
        print("Model Registered...")
        

register_models()
