import numpy as np

def mean_squared_error(target, output):
    r"""
    Calculate the Mean Squared Error (MSE) between the target and output signals.

    The Mean Squared Error is defined as:

        MSE = (1/N) * Σ (y_i - ŷ_i)^2

    where y_i is the target and ŷ_i is the predicted/output value.

    Args:
        target (np.array or list): The ground truth target signal.
        output (np.array or list): The predicted or simulated output signal.

    Returns:
        float: The MSE between the target and output signals.
    """
    length = len(target)
    return np.sum((target - output) ** 2) / length


def nmse_calculate(measured, predicted):
    r"""
    Calculate the Normalized Mean Squared Error (NMSE) between measured and predicted signals.

    NMSE is defined as:
    NMSE = MSE / (mean_measured * mean_predicted)

        Where:
            mean_measured  = average of the measured (simulation) signal
            mean_predicted = average of the predicted (target) signal
            
    Args:
        measured (np.array or list): The measured or simulated signal.
        predicted (np.array or list): The predicted (target) signal.

    Returns:
        float: The NMSE between measured and predicted signals.
    """
    predicted = np.array(predicted)
    measured = np.array(measured)

    mse = np.mean((predicted - measured) ** 2)
    denom = np.mean(predicted) * np.mean(measured)

    return mse / denom
