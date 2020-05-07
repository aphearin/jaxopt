"""
"""
from jax import random as jax_random
from jax import numpy as jax_np


def read_init_data():
    """Read data from disk and prepare for measurement."""
    pass


def get_init_params():
    """Get initial parameters of the model."""
    pass


def get_target_data(init_data, ran_key):
    """Make the measurement on the input data.
    Return the target data, and any metadata
    that will be needed to compute the loss.
    """
    old_key, ran_key = jax_random.split(ran_key)

    measurement_data = {}
    metadata = {}
    return target_data_vector, metadata, ran_key


def get_prediction_data(metadata_target, ran_key):
    """Get the metadata required to predict the target data.
    """
    old_key, ran_key = jax_random.split(ran_key)
    prediction_data = metadata_target
    return prediction_data, ran_key


def get_mse_loss_data(data_prediction, data_target, metadata_target, ran_key):
    """
    """
    mse_loss_data = data_prediction, data_target, metadata_target
    return mse_loss_data, ran_key


def predict_target(params, prediction_data):
    """Get model predictions to compare to target."""
    return predictions


def calculate_mse_loss(params, mse_loss_data):
    prediction_data = mse_loss_data[0]
    predictions = predict_target(params, prediction_data)
    targets = mse_loss_data[1]

    loss = 0.0
    for target, pred in zip(targets, predictions):
        loss += _jax_mse(target, pred)

    return loss


def _jax_mse(x, y):
    d = y - x
    return jax_np.sum(d * d)


def _resample_target_data(ran_key, resample_rate):
    ran_key, _ran_key = jax_random.split(ran_key)
    uran = jax_random.uniform(_ran_key)
    if uran < resample_rate:
        resample = True
    else:
        resample = False
    return resample, ran_key
