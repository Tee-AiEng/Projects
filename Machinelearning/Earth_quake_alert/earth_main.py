from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib
import pandas as pd
import numpy as np

# intantiate your app
app = FastAPI()

# load your model and and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# create a pydantic model for enforcing
class winefeatures(BaseModel):
    magnitude:float
    depth:float
    cdi :float
    mmi:float
    sig:float


#lets create our endpoint
@app.get("/")
def home():
    return {
        "Message":"Welcome to Tee Earth Quake Alert"
    }


@app.post("/prediction")
def get_predictions(wine:winefeatures):
    # convert features to 2darray
    features = np.array([[
    wine.magnitude,
     wine.depth,
     wine.cdi,
      wine.mmi,
     wine.sig
    ]])


    # scale your features
    scaled_features = scaler.transform(features)

    #get prediction 
    predictions = model.predict(scaled_features)
    
    return {"Earth_ quake_Alert":str(predictions[0])}

#let run our app
# uvicorn earth_main:app --reload
