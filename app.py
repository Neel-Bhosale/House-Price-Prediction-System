from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load the saved model with error handling
try:
    with open("best_xgb_model.pkl", "rb") as file:
        model = pickle.load(file)
    print(" Model loaded successfully!")
except Exception as e:
    print(f" Error loading model: {e}")
    model = None  # Prevent crashing if model load fails

# Define request schema
class PredictionInput(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity_INLAND: int
    ocean_proximity_ISLAND: int
    ocean_proximity_NEAR_BAY: int
    ocean_proximity_NEAR_OCEAN: int

# Define prediction endpoint
@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        if model is None:
            return {"error": "Model is not loaded."}

        # Convert input to numpy array with correct dtype
        features = np.array([[input_data.longitude, input_data.latitude, input_data.housing_median_age,
                              input_data.total_rooms, input_data.total_bedrooms, input_data.population,
                              input_data.households, input_data.median_income,
                              input_data.ocean_proximity_INLAND, input_data.ocean_proximity_ISLAND,
                              input_data.ocean_proximity_NEAR_BAY, input_data.ocean_proximity_NEAR_OCEAN]], dtype=np.float32)

        # Make prediction
        prediction = model.predict(features)[0]

        return {"predicted_price": float(prediction)}  # Ensure JSON serializable output

    except Exception as e:
        return {"error": str(e)}
