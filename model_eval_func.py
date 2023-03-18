# library imports
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score, max_error


def model_eval_func(y_pred,
                    y):
    assert len(y) == len(y_pred)
    arr = [metric(y_pred, y) for metric in [r2_score, mean_absolute_error, mean_squared_error, explained_variance_score, max_error]]
    arr.append(min(y))
    arr.append(max(y))
    arr.append(np.mean(y))
    arr.append(np.std(y))
    return ",".join(arr)


def model_eval_func_titles():
    return ",".join(["r2_score",
                     "mean_absolute_error",
                     "mean_squared_error",
                     "explained_variance_score",
                     "max_error",
                     "min_y",
                     "max_y",
                     "mean_y"])
