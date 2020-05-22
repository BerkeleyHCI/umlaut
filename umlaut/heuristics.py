import numpy as np
import tensorflow as tf
from umlaut.errors import InputNotFloatingError
from umlaut.errors import InputNotNormalizedError
from umlaut.errors import NaNInLossError


def run_pretrain_heuristics(model, x_train):
    NotImplemented

def run_epoch_heuristics(epoch, model, logs, x_train):
    errors_raised = []
    errors_raised.append(check_input_normalization(epoch, x_train))
    errors_raised.append(check_input_is_floating(epoch, model, x_train))
    errors_raised.append(check_nan_in_loss(epoch, logs['loss']))
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


def check_input_is_floating(epoch, model, x_train):
    '''Returns an `InputNotFloatingError` if input is not floating.
    '''
    # x_train is a numpy object, not a tensor
    if not tf.as_dtype(x_train.dtype).is_floating:
        return InputNotFloatingError(epoch)


def check_nan_in_loss(epoch, loss):
    '''Returns a NanInLossError if loss is NaN.
    '''
    if np.isnan(loss):
        return NaNInLossError(epoch)