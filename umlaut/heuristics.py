import numpy as np
from umlaut.errors import InputNotNormalizedError

def run_heuristics(epoch, x_train):
    errors_raised = []
    errors_raised.append(check_input_normalization(epoch, x_train))
    return errors_raised

def check_input_normalization(epoch, x_train, bounds=(-1, 1)):
    '''Returns an `InputNotNormalizedError` if inputs exceed bounds.
    '''
    x_min = np.min(x_train)
    x_max = np.max(x_train)
    remark = ''
    if x_min < bounds[0]:
        remark = remark + f'x_min is {x_min}, less than {bounds[0]}. '
    if x_max > bounds[1]:
        remark = remark + f'x_max is {x_max}, greater than {bounds[1]}.'
    if remark:
        return InputNotNormalizedError(epoch, remark)