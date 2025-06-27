import os
import json
import joblib
import pandas as pd
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sklearn import set_config
from mlflow import MlflowClient
from dotenv import load_dotenv

from scripts.data_clean_utils import perform_data_cleaning

# ✅ Set output of sklearn to pandas DataFrame
set_config(transform_output='pandas')

# ✅ Load .env variables (like DAGSHUB_TOKEN)
load_dotenv()

# ✅ Authenticate DagsHub
import dagshub
from dagshub.auth import add_app_token

token = os.getenv("DAGSHUB_TOKEN")
if token:
    print("✅ DAGSHUB_TOKEN found")
    add_app_token(repo_url="https://dagshub.com", token=token)
else:
    raise ValueError("❗ DAGSHUB_TOKEN not found in environment variables.")

# ✅ Initialize DagsHub with MLflow tracking
dagshub.init(
    repo_owner='mepaluttam',
    repo_name='swiggy-delivery-time-predicion',
    mlflow=True
)
mlflow.set_tracking_uri("https://dagshub.com/mepaluttam/swiggy-delivery-time-predicion.mlflow")

# ✅ Pydantic input schema
class Data(BaseModel):  
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: str
    Delivery_person_Ratings: str
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: str
    Festival: str
    City: str

# ✅ Helper functions
def load_model_information(file_path):
    with open(file_path) as f:
        return json.load(f)

def load_model(model_path):
    return joblib.load(model_path)

# ✅ Load model and preprocessor
model_info = load_model_information("run_information.json")
model_name = model_info['model_name']
stage = "Production"

client = MlflowClient()
latest_model_ver = client.get_latest_versions(name=model_name, stages=[stage])
print(f"✅ Latest model in Production: version {latest_model_ver[0].version}")

model_path = f"models:/{model_name}/{stage}"
model = mlflow.sklearn.load_model(model_path)

preprocessor = load_model("models/preprocessor.joblib")

model_pipe = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', model)
])

# ✅ FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return "Welcome to the Swiggy Food Delivery Time Prediction App"

@app.post("/predict")
def do_predictions(data: Data):
    input_df = pd.DataFrame({
        'ID': data.ID,
        'Delivery_person_ID': data.Delivery_person_ID,
        'Delivery_person_Age': data.Delivery_person_Age,
        'Delivery_person_Ratings': data.Delivery_person_Ratings,
        'Restaurant_latitude': data.Restaurant_latitude,
        'Restaurant_longitude': data.Restaurant_longitude,
        'Delivery_location_latitude': data.Delivery_location_latitude,
        'Delivery_location_longitude': data.Delivery_location_longitude,
        'Order_Date': data.Order_Date,
        'Time_Orderd': data.Time_Orderd,
        'Time_Order_picked': data.Time_Order_picked,
        'Weatherconditions': data.Weatherconditions,
        'Road_traffic_density': data.Road_traffic_density,
        'Vehicle_condition': data.Vehicle_condition,
        'Type_of_order': data.Type_of_order,
        'Type_of_vehicle': data.Type_of_vehicle,
        'multiple_deliveries': data.multiple_deliveries,
        'Festival': data.Festival,
        'City': data.City
    }, index=[0])

    cleaned_data = perform_data_cleaning(input_df)
    prediction = model_pipe.predict(cleaned_data)[0]
    return {"predicted_delivery_time": prediction}

# ✅ Optional: Only for local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
