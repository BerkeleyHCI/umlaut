import inspect
import numpy as np
import tensorflow as tf

import umlaut.errors


def run_pretrain_heuristics(model, source_module):
    errors_raised = []
    errors_raised.append(check_softmax_computed_before_loss(model))
    return errors_raised

def run_epoch_heuristics(epoch, model, logs, x_train, source_module):
    errors_raised = []
    errors_raised.append(check_input_normalization(epoch, x_train))
    errors_raised.append(check_input_is_floating(epoch, model, x_train))
    errors_raised.append(check_nan_in_loss(epoch, logs))
    errors_raised.append(check_overfitting(epoch, model, logs))
    return errors_raised


def check_input_normalization(epoch, x_train):
    '''Returns an `InputNotNormalizedError` if inputs exceed bounds.
    '''
    x_min = np.min(x_train)
    x_max = np.max(x_train)
    remark = ''
    if x_min < -1:
        remark = remark + f' The minimum input value is {x_min}, less than the typical value of -1.'
    if x_max > 1:
        remark = remark + f' The maximum input value is {x_max}, greater than the typical value of 1.'
    if remark:
        return umlaut.errors.InputNotNormalizedError(epoch, remark)


def check_input_is_floating(epoch, model, x_train):
    '''Returns an `InputNotFloatingError` if input is not floating.
    '''
    # x_train is a numpy object, not a tensor
    if not tf.as_dtype(x_train.dtype).is_floating:
        return umlaut.errors.InputNotFloatingError(epoch)


def check_nan_in_loss(epoch, logs):
    '''Returns a NanInLossError if loss is NaN.
    '''
    loss = logs['loss']
    if np.isnan(loss):
        return umlaut.errors.NaNInLossError(epoch)


def check_softmax_computed_before_loss(model):
    '''Ensures the loss function used has a proper from_logits setting.
    '''
    # from_logits is always False by default per source code
    from_logits = False
    if issubclass(type(model.loss), tf.keras.losses.Loss):
        # get from_logits arg from Loss class family
        from_logits = model.loss._fn_kwargs.get('from_logits', False)
    last_layer_is_softmax = isinstance(model.layers[-1], tf.keras.layers.Softmax)
    if not last_layer_is_softmax and not from_logits:
        return umlaut.errors.NoSoftmaxActivationError()


def check_learning_rate_range(epoch, model):
    NotImplemented


def check_overfitting(epoch, model, logs):
    if not model.history.history:
        return
    last_loss = model.history.history['loss'][-1]
    last_val_loss = model.history.history['val_loss'][-1]
    d_loss = logs['loss'] - last_loss
    d_val_loss = logs['val_loss'] - last_val_loss
    if d_val_loss > 0:
        if d_loss <= 0:
            return umlaut.errors.OverfittingError(epoch)


def check_high_validation_acc(epoch, logs):
    NotImplemented


def check_initialization(epoch, model, logs):
    NotImplemented
