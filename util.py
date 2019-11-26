"""
    util
    ~~~~~~~~~~~~~
    Implements various utility functions

"""

import constants
import numpy as np
import pandas as pd

def get_dataframes(filePath, yname="rating"):
    dataframe = pd.read_csv(filePath)

    if not yname:
        return dataframe

    y = dataframe[yname]
    x = dataframe.drop(yname, axis=1)

    return x, y

def get_training_data():
    return get_dataframes(constants.TRAIN_FILE)

def get_validation_data():
    return get_dataframes(constants.VAL_FILE)

def get_test_data():
    return get_dataframes(constants.TEST_FILE, yname="")

def compute_mse(predicted_y, y):
    mse = np.sum((predicted_y - y) ** 2) / predicted_y.shape[0]
    return mse

# todo: some function that returns a dataframe ready to input test answers