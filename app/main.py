# import FastAPI from fastapi, JSONResponse from starlette.responses, load from joblib and pandas

from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd


# Instating the fast API class
app = FastAPI()

# Load rf_1_pipeline pipeline

rf_1_pipeline = load('../models/rf_1_pipeline.joblib')


# create a function called `read_root()` that will return a 
# dictionary with `Hi` as key and `There` as value. Add a decorator to it in order to add a GET endpoint to `app` on the root

@app.get("/")
def read_root():
    return {"""
    Displaying a brief description of the project objectives, 
    list of endpoints, expected input parameters and output format of the model,
     link to the Github repo related to this project

    """}






@app.get('/health', status_code=200)
def healthcheck():
    return 'Predictor is all ready to go!'



"""
create a function called `format_features()`
with `item`,`store`, and `date` 
as input parameters that will return a dictionary with the names of the features as keys and the inpot parameters as lists
"""

def format_features(
 
    item:str,
    store:str,
    date:str
    ):
    return {
        'item': [item],
        'store': [store],
        'date': [date],
        
    }




@app.get("/cvd/risks/prediction")
def predict(
    item:str,
    store:str,
    date:str
):
    features = format_features(
        item,
        store,
        date
        )
    obs = pd.DataFrame(features)
    pred = rf_1_pipe.predict(obs)
    return JSONResponse(pred.tolist())



    