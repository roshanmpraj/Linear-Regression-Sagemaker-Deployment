"""
inference.py

Defines model loading and inference logic for a SageMaker endpoint.
"""

import json
import joblib
import numpy as np
import os


def model_fn(model_dir):
    """
    Load the trained model from the model directory.
    """
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, content_type):
    """
    Parse incoming JSON request.
    """
    if content_type == "application/json":
        return np.array(json.loads(request_body))
    raise ValueError("Unsupported content type")


def predict_fn(input_data, model):
    """
    Run model prediction.
    """
    return model.predict(input_data)


def output_fn(prediction, content_type):
    """
    Convert prediction output to JSON.
    """
    return json.dumps(prediction.tolist())
