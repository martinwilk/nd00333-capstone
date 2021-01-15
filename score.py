# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import os
import pickle
import numpy as np
import pandas as pd
import joblib
from azureml.core.model import Model



from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame({"fixed acidity": pd.Series([0.0], dtype="float64"), "volatile acidity": pd.Series([0.0], dtype="float64"), "citric acid": pd.Series([0.0], dtype="float64"), "residual sugar": pd.Series([0.0], dtype="float64"), "chlorides": pd.Series([0.0], dtype="float64"), "free sulfur dioxide": pd.Series([0.0], dtype="float64"), "total sulfur dioxide": pd.Series([0.0], dtype="float64"), "density": pd.Series([0.0], dtype="float64"), "pH": pd.Series([0.0], dtype="float64"), "sulphates": pd.Series([0.0], dtype="float64"), "alcohol": pd.Series([0.0], dtype="float64"), "is_red_wine": pd.Series([0.0], dtype="float64")})
output_sample = np.array([0])


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = Model.get_model_path('wine-quality-model')
    model = joblib.load(model_path)
    


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
