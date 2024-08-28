from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('logistic_regression_model.pkl')

# Initialize FastAPI app
app = FastAPI()

# Define the request body model
class PredictionRequest(BaseModel):
    Pclass: int
    Sex: int
    Age: int
    SibSp: int
    Parch: int
    Fare: float
    Embarked: int

# Define the prediction route
@app.post('/predict')
def predict_survival(data: PredictionRequest):
    # Convert the incoming data to a DataFrame
    df = pd.DataFrame([data.dict().values()], columns=data.dict().keys())
    
    # Make the prediction
    prediction = model.predict(df)
    
    # Return the result
    return {"prediction": int(prediction[0])}

# Define a root route for sanity check
@app.get('/')
def read_root():
    return {"message": "Welcome to the Titanic Survival Prediction API"}
