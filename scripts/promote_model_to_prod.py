import mlflow
from mlflow import MlflowClient
import json
import os

# === Set the MLflow Tracking URI (DagsHub) ===
mlflow.set_tracking_uri("https://dagshub.com/mepaluttam/swiggy-delivery-time-predicion.mlflow")

# === Optionally validate credentials ===
mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")

if not (mlflow_username and mlflow_password):
    raise ValueError("❗❗❗ MLFLOW_TRACKING_USERNAME and/or MLFLOW_TRACKING_PASSWORD not set.")

# === Load model name from run_information.json ===
def load_model_information(file_path):
    with open(file_path) as f:
        return json.load(f)

model_info = load_model_information("run_information.json")
model_name = model_info["model_name"]
stage = "Staging"
promotion_stage = "Production"

# === Initialize MLflow client ===
client = MlflowClient()

# === Fetch the latest version in 'Staging' ===
latest_versions = client.get_latest_versions(name=model_name, stages=[stage])

if not latest_versions:
    raise ValueError(f"No model versions found in stage: {stage}")

latest_model_version = latest_versions[0].version

# === Promote model to 'Production' ===
client.transition_model_version_stage(
    name=model_name,
    version=latest_model_version,
    stage=promotion_stage,
    archive_existing_versions=True
)

print(f"✅ Successfully promoted model '{model_name}' version {latest_model_version} to '{promotion_stage}'.")
