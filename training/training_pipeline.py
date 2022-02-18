import os
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, StepSequence
from azureml.core import Workspace, Experiment, RunConfiguration
from azureml.core.environment import CondaDependencies
from azureml.core.authentication import ServicePrincipalAuthentication

sp = ServicePrincipalAuthentication(tenant_id=os.environ['AML_TENANT_ID'],
                                    service_principal_id=os.environ['AML_PRINCIPAL_ID'],
                                    service_principal_password=os.environ['AML_PRINCIPAL_PASS'])
ws = Workspace.get(name=os.environ["WORKSPACE_NAME"],
               subscription_id=os.environ["SUBSCRIPTION_ID"],
               resource_group=os.environ["RESOURCE_GROUP"],
               auth=sp)


run_config = RunConfiguration()
run_config.environment.python.conda_dependencies = CondaDependencies.create(python_version="3.8",
                                                                            pip_packages=["numpy", "pandas",
                                                                                          "scikit-learn", "azureml-core",
                                                                                          "azureml-defaults", "azureml-pipeline"])


train_prepped_data = OutputFileDatasetConfig("train_prepped")
test_prepped_data = OutputFileDatasetConfig("test_prepped")

prep_step = PythonScriptStep(name="prepare-data",
                         source_directory="training/training_pipeline",
                         script_name="prep.py",
                         compute_target="aml-cluster",
                         arguments = ["--train-ds-name", "wine-quality-train",
                                      "--test-ds-name", "wine-quality-test",
                                      "--train-out-folder", train_prepped_data,
                                      "--test-out-folder", test_prepped_data],
                        runconfig=run_config,
                        allow_reuse=False)
train_step = PythonScriptStep(name="train-model",
                        source_directory="training/training_pipeline", 
                        script_name="train.py",
                        compute_target="aml-cluster",
                        arguments=["--train-data", train_prepped_data.as_input(),
                                   "--test-data", test_prepped_data.as_input()],
                        runconfig=run_config)

register_step = PythonScriptStep(name="register-model",
                        source_directory="registering", 
                        script_name="register_model.py",
                        compute_target="aml-cluster",
                        arguments = ["--model-name", "wine-quality-lr"],
                        runconfig=run_config)

step_sequence = StepSequence(steps=[prep_step, train_step, register_step])
pipeline_steps = Pipeline(workspace=ws, steps=step_sequence)
experiment = Experiment(name="wine-quality-training", workspace=ws)
run = experiment.submit(pipeline_steps)
run.wait_for_completion(show_output=True)

print("publish pipeline")
# Publish pipeline 
published_pipeline = run.publish_pipeline(
                             name="wine-quality-training-pipeline",
                             description="Training pipeline for wine-quality-app",
                             version="1.0")
print("starting schedule")
# Create schedule
from azureml.pipeline.core import ScheduleRecurrence, Schedule
minutely = ScheduleRecurrence(frequency='Minute', interval=5)
pipeline_schedule = Schedule.create(ws, name='continual training',
                                        description='continual training wine app',
                                        pipeline_id=published_pipeline.id,
                                        experiment_name='wine-app-training-schedule',
                                        recurrence=minutely)


