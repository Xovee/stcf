from sklearn.metrics import mean_squared_error


def get_MSE(pred, real):
    return mean_squared_error(real.flatten(), pred.flatten())

def print_metrics(pred, real):
    mse = get_MSE(pred, real)

    print('Test: MSE={:.6f}'.format(mse))
