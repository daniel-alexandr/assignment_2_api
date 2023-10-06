# import FastAPI from fastapi, JSONResponse from starlette.responses, load from joblib and pandas

from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
from datetime import datetime, timedelta


# Instating the fast API class
app = FastAPI()

# Load rf_1_pipeline pipeline

rf_1_pipeline = load('../models/rf_1_pipeline.joblib')
ARIMA_model= load('../models/ARIMA_model.joblib')


# create a function called `read_root()` that will return a 
# description

@app.get("/")
def read_root():
    return {"""
    This project objective is to help the company company to do inventory management, and to set the target for each stores to adopt.
    On top of that, we also want to help the finance team to do cash flow management, planning and budgeting.
    Those objective are addressed by:

    1. Predictive model to predict revenue for a given inputted item in specific inputted store on a specific inputted date, delivered via API. 
    2. Forecasting model to forecast the total revenue for next 7 days from a given inputted date, delivered via API

    List of endpoints:
    1. '/health'  : Checking whether connection is good to go
    
    2. '/sales/stores/item/' : API endpoint to use the predictive model. This endpoint will output the revenue prediction for an item in a store on a date
    
        Expected inputs are: 
        item_id: Only enter one from this list []
        store_id: Only enter one from this list [CA_1]
        Date: Has to be in a following format 'yyyy-mm-dd' and a valid factual date. 

    3. '/sales/national/': API endpoint to use the forecasting model. This endpoint will output the forecast for the next 7 days from a inputted starting date.

        Date: Has to be in a following format 'yyyy-mm-dd' and a valid factual date. 


    Link to GitHub:

    Model training : https://github.com/daniel-alexandr/Assignment2_ML_as_a_service
    API backend: https://github.com/daniel-alexandr/assignment_2_api

    """}





@app.get('/health', status_code=200)
def healthcheck():
    return 'Predictor is all ready to go!'



"""
create a function called `format_features()`
with `item`,`store`, and `date` 
as input parameters that will return a dictionary with the names of the features as keys and the input parameters as lists
"""

def format_features(
 
    item:str,
    store:str,
    date:str
    ):
    

    # Define the starting date (January 29, 2011)
    # starting_date = datetime(2011, 1, 29)

    # Define your event date
    inputted_date = datetime.strptime(date, '%Y-%m-%d')  # Change this to your event date
    event_date=inputted_date - timedelta(days=28)

    # Extract the week number from the datetime object
    week_number = int(event_date.strftime('%U'))

    # Define logic for limit value
    if week_number > 53:
        week_number = 53
    elif week_number == 0:
        week_number=1

    # Format the week number to be double digit string
    week_number = "{:02d}".format(week_number)

    # Splitting the item_id to get cat_id
    cat_id=item_id.split('_')[0]


    return {
        'cat_id': [cat_id],
        'store_id': [store],
        'week': [week_number]
        
    }


"""
Create a forecast_7_days function that will return list of forecasted 
values of 7 days from the next day of a given date

"""
def forecast_7_days(
    date:str
):
    preds=[]
    inputted_date = datetime.strptime(date, '%Y-%m-%d')
    for index in range(1,8):
        pred=ARIMA_model.predict(inputted_date + timedelta(days=index))[0]
        preds.append(pred)
    return preds






@app.get("/sales/stores/item/")
def predict(
    item_id:str,
    store_id:str,
    date_yymmdd:str
):
    features = format_features(
        item_id,
        store_id,
        date_yymmdd
        )
    obs = pd.DataFrame(features)
    pred = rf_1_pipeline.predict(obs)
    return JSONResponse(pred.tolist())





@app.get("/sales/national/")
def forecast(
    date_yymmdd:str
):
    preds = forecast_7_days(date_yymmdd)
    total_7_days= sum(preds)
    return JSONResponse(total_7_days.tolist())




