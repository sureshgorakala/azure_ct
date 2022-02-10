import os
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline
from azureml.core import Workspace, Experiment, RunConfiguration
from azureml.core.environment import CondaDependencies
run_config = RunConfiguration()
run_config.environment.python.conda_dependencies = CondaDependencies.create(python_version="3.8",
                                                                            pip_packages=["numpy", "pandas",
                                                                                          "scikit-learn", "azureml-core",
                                                                                          "azureml-defaults", "azureml-pipeline"])
ws = Workspace.get(name=os.environ["WORKSPACE_NAME"],
               subscription_id=os.environ["SUBSCRIPTION_ID"],
               resource_group=os.environ["RESOURCE_GROUP"])

train_prepped_data = OutputFileDatasetConfig("train_prepped")
test_prepped_data = OutputFileDatasetConfig("test_prepped")

step1 = PythonScriptStep(name="prepare-data",
                         source_directory="training/training_pipeline",
                         script_name="prep.py",
                         compute_target="aml-cluster",
                         arguments = ["--train-ds-name", "wine-quality-train",
                                      "--test-ds-name", "wine-quality-test",
                                      "--train-out-folder", train_prepped_data,
                                      "--test-out-folder", test_prepped_data],
                        runconfig=run_config,
                        allow_reuse=False)
step2 = PythonScriptStep(name="train-model",
                        source_directory="training/training_pipeline", 
                        script_name="train.py",
                        compute_target="aml-cluster",
                        arguments=["--train-data", train_prepped_data.as_input(),
                                   "--test-data", test_prepped_data.as_input()],
                        runconfig=run_config)
pipeline_steps = Pipeline(workspace=ws, steps=[step1, step2])
experiment = Experiment(name="wine-quality-training", workspace=ws)
run = experiment.submit(pipeline_steps)
run.wait_for_completion(show_output=True)
