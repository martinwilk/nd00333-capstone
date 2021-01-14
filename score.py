import json
import joblib
import pandas as pd
from azureml.core.model import Model

# Called when the service is loaded
def init():
    global model
    # Get the path to the registered model file and load it
    model_path = Model.get_model_path('wine-quality-model')
    model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    data = pd.DataFrame.from_dict(json.loads(raw_data)['data'])
    # Get a prediction from the model
    predictions = model.predict(data)
    # Return the predictions as any JSON serializable format
    return {"result": pd.Series(predictions).replace({-1: "BAD", 0:"MEDIUM", 1: "GOOD"}).tolist()}
