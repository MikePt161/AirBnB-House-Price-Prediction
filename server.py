# Import the web framework
from fastapi import FastAPI
# Import the server
import uvicorn
# This is a library that does data validation
from pydantic import BaseModel

from joblib import load
import pandas as pd
import numpy as np
from preprocessing import impute_missing_values, scale_features, fix_room_type, one_hot_amenities, \
    one_hot_neighbourhood, fix_rates, fix_bools, fix_neighbourhoods, fix_amenities, create_bathrooms_number

# Start the app
app = FastAPI()

# Specify the input json format as a dict with all the features of the model
class ListingData(BaseModel):
    neighbourhood_cleansed: str
    host_response_rate: str
    host_acceptance_rate: str
    host_is_superhost: str
    host_listings_count: float
    host_identity_verified: str
    latitude: float
    longitude: float
    room_type: str
    bathrooms_text: str
    bedrooms: float
    beds: float
    accommodates: int
    amenities: str
    number_of_reviews: int
    reviews_per_month: float


# Specify the input json format as an dict with a key and a value of an array
class ListingDataArray(BaseModel):
    data: list

# This function runs when you start the server and is responsible for loading the model
@app.on_event('startup')
def load_model():
    global rf
    rf = load('data/rf.joblib')

# The following @app.get('/blah') or app.post('/blah')
# specify endpoints or url paths that can be accessed with GET or POST requests
# Usuall GET is getting information for the server, whereas POST sends some data in the server
# the server does some processing, or saving of that data and returns something

# This is a health check endpoint that assures that the server is running. Usually for monitoring purposes
@app.get('/health')
def health():
    return {'status': 'ready'}

# This endpoint just shows the importances of the features of the RandomForest Classifier.
# This can be used to be sure that we have the correct model loaded, as well as provide
# a visualization/justification of the important features for our model. Not all ML models
# have this attribute
# @app.get('/lm_coefs')
# def get_coefs():
#     print('hi')
#     coefs = {}
#     for n, imp in sorted(zip(rf.feature_names, rf.coef_), key= lambda x: x[1], reverse=True):
#         coefs[n] = imp
    
#     return coefs

@app.post('/predict_list')
def predict_list(b: ListingDataArray):
    new_samples = b.data
    new_samples = pd.DataFrame.from_records(new_samples)
    new_samples = preprocess_input(new_samples)
    res = rf.predict(new_samples)
    return {"Predictions": {f"Prediction of sample {i}": pred for i, pred in enumerate(res)}}


@app.post('/predict')
def predict(b: ListingData):
    new_sample = pd.DataFrame(b.dict(), index=[0])
    new_sample = preprocess_input(new_sample)

 
    res = rf.predict(new_sample)

    return {"prediction": res.item(0)}



def preprocess_input(data):
    data = data.replace('', np.nan)

    data = fix_rates(data)
    data = fix_bools(data)
    data = fix_neighbourhoods(data)
    data = fix_amenities(data)
    data = create_bathrooms_number(data)
    data = impute_missing_values(data)
    data = scale_features(data)
    data = fix_room_type(data)
    data = one_hot_amenities(data)
    data = one_hot_neighbourhood(data)

    return data




if __name__ == '__main__':
    uvicorn.run("server:app", host="127.0.0.1", port=8000, log_level="info", reload=True)