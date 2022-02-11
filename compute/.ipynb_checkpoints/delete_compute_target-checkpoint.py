import os
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.authentication import ServicePrincipalAuthentication

sp = ServicePrincipalAuthentication(tenant_id=os.environ['AML_TENANT_ID'],
                                    service_principal_id=os.environ['AML_PRINCIPAL_ID'],
                                    service_principal_password=os.environ['AML_PRINCIPAL_PASS'])
ws = Workspace.get(name=os.environ["WORKSPACE_NAME"],
               subscription_id=os.environ["SUBSCRIPTION_ID"],
               resource_group=os.environ["RESOURCE_GROUP"],
               auth=sp)

cluster_name = "aml-cluster"

print("Deleting compute cluster")
ws.compute_targets[cluster_name].delete()