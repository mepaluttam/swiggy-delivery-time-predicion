import pytest
import mlflow
from mlflow import MlflowClient
import dagshub
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Optionally check the token exists
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if dagshub_token:
    import dagshub
    dagshub.init(
        repo_owner='mepaluttam',
        repo_name='swiggy-delivery-time-predicion',
        mlflow=True
    )
else:
    print("⚠️ DAGSHUB_TOKEN not set — skipping dagshub.init()")

# set the mlflow tracking server
mlflow.set_tracking_uri("https://dagshub.com/mepaluttam/swiggy-delivery-time-predicion.mlflow")


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info
# set model name
model_name = load_model_information("run_information.json")["model_name"]
@pytest.mark.parametrize(argnames="model_name, stage",
                         argvalues=[(model_name, "Staging")])
def test_load_model_from_registry(model_name,stage):
    client = MlflowClient()
    latest_versions = client.get_latest_versions(name=model_name,stages=[stage])
    latest_version = latest_versions[0].version if latest_versions else None
    
    assert latest_version is not None, f"No model at {stage} stage"
    
    # load the model
    model_path = f"models:/{model_name}/{stage}"
    # load the latest model from model registry
    model = mlflow.sklearn.load_model(model_path)
    
    assert model is not None, "Failed to load model from registry"
    print(f"The {model_name} model with version {latest_version} was loaded successfully")
    