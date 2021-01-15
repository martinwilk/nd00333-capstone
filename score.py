import json
import joblib
import pandas as pd
import numpy as np
from azureml.core.model import Model
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

# Called when the service is loaded
def init():
    global model
    # Get the path to the registered model file and load it
    model_path = Model.get_model_path('wine-quality-model')
    model = joblib.load(model_path)
    
input_sample = pd.DataFrame(data=[{
             "fixed acidity":7.90000,
             "volatile acidity":0.52000,
             "citric acid":0.26000, 
             "residual sugar":2.2000,
             "chlorides":0.07900,
             "free sulfur dioxide":14.0000,
             "total sulfur dioxide":38.0000,
             "density":0.99675,
             "pH":3.31000,
             "sulphates":0.62000,
             "alcohol":10.20000,
             "is_red_wine": 1            
            }])

output_sample = np.array(["MEDIUM"])             

@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))

# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    data = pd.DataFrame.from_dict(json.loads(raw_data)['data'])
    # Get a prediction from the model
    predictions = model.predict(data)
    # Return the predictions as any JSON serializable format
    return json.dumps({"result": pd.Series(predictions).replace({-1: "BAD", 0:"MEDIUM", 1: "GOOD"}).tolist()})
