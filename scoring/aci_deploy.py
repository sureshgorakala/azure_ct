from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.model import Model
from azureml.core import Workspace
from azureml.core.environment import Environment
env = Environment.from_conda_specification(name="service-env", file_path="scoring/scoring/env.yaml")

model_inference_config = InferenceConfig(source_directory="scoring/scoring",
                                        entry_script="scoring.py",
                                        environment=env)
model_inference_config.validate_configuration()
ws = Workspace.get(name=os.environ["WORKSPACE_NAME"],
               subscription_id=os.environ["SUBSCRIPTION_ID"],
               resource_group=os.environ["RESOURCE_GROUP"])
aci_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)
aci_config.validate_configuration()

model2 = Model(ws, name="wine-quality-lr")
model1 = Model(ws, name="wine-quality-scaler")
service = Model.deploy(ws, "wine-quality-aci", [model1, model2], model_inference_config, aci_config)
service.wait_for_deployment(show_output = True)
print(service.state)

service.get_logs()
