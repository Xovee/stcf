import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_log_error


def get_MSE(pred, real):
    return mean_squared_error(real.flatten(), pred.flatten())


def get_MAE(pred, real):
    return mean_absolute_error(real.flatten(), pred.flatten())


def get_MAPE(pred, real):
    return mean_absolute_percentage_error(real.flatten()+1, pred.flatten()+1)


def get_MSLE(pred, real):
    return mean_squared_log_error(real.flatten()+1, pred.flatten()+1)


def get_ACC(pred, real):
    return np.mean(np.abs(((real.flatten()+1) - (pred.flatten()+1)) / (real.flatten()+1)) <= .2) * 100


def print_metrics(pred, real):
    mse = get_MSE(pred, real)

    # feel free to print other metrics here
    print('Test: MSE={:.6f}'.format(mse))
