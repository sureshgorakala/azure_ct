from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.model import Model
from azureml.core import Workspace
from azureml.core.environment import Environment
import os
from azureml.core.authentication import ServicePrincipalAuthentication

sp = ServicePrincipalAuthentication(tenant_id=os.environ['AML_TENANT_ID'],
                                    service_principal_id=os.environ['AML_PRINCIPAL_ID'],
                                    service_principal_password=os.environ['AML_PRINCIPAL_PASS'])
ws = Workspace.get(name=os.environ["WORKSPACE_NAME"],
               subscription_id=os.environ["SUBSCRIPTION_ID"],
               resource_group=os.environ["RESOURCE_GROUP"],
               auth=sp)

env = Environment.from_conda_specification(name="service-env", file_path="scoring/scoring/env.yaml")

model_inference_config = InferenceConfig(source_directory="scoring/scoring",
                                        entry_script="scoring.py",
                                        environment=env)
model_inference_config.validate_configuration()

aci_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)
aci_config.validate_configuration()

model2 = Model(ws, name="wine-quality-lr")
model1 = Model(ws, name="wine-quality-scaler")
print(model2, model1)

service = Model.deploy(ws, "wine-quality-aci", [model1, model2], 
                       model_inference_config, aci_config, 
                       overwrite=True)
service.wait_for_deployment(show_output = True)
print(service.state)

service.get_logs()