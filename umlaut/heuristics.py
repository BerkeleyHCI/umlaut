import inspect
import numpy as np
import tensorflow as tf

import umlaut.errors

def _get_acc_key(logs, val=False):
    key = ''
    if val:
        key = 'val_'
    if logs.get(key + 'acc', False):
        return key + 'acc'
    return key + 'accuracy'


def run_pretrain_heuristics(model, source_module):
    errors_raised = []
    errors_raised.append(check_softmax_computed_before_loss(model))
    return errors_raised


def run_epoch_heuristics(epoch, model, logs, x_train, source_module):
    errors_raised = []
    errors_raised.append(check_input_normalization(epoch, x_train, source_module))
    errors_raised.append(check_input_is_floating(epoch, model, x_train))
    errors_raised.append(check_nan_in_loss(epoch, logs))
    errors_raised.append(check_overfitting(epoch, model, logs))
    errors_raised.append(check_high_validation_acc(epoch, model, logs))
    return errors_raised


def check_accuracy_is_added_to_metrics(logs, source_module):
    NotImplemented


def check_validation_is_added_to_fit(logs, source_module):
    NotImplemented


def check_input_normalization(epoch, x_train, source_module):
    '''Returns an `InputNotNormalizedError` if inputs exceed bounds.
    '''
    x_min = np.min(x_train)
    x_max = np.max(x_train)
    remark = ''
    if x_min < -1:
        remark = remark + f'The minimum input value is {x_min}, less than the typical value of -1.'
    if x_max > 1:
        remark = remark + f'The maximum input value is {x_max}, greater than the typical value of 1.'
    if remark:
        return umlaut.errors.InputNotNormalizedError(epoch, remark, source_module['path'])


def check_input_is_floating(epoch, model, x_train):
    '''Returns an `InputNotFloatingError` if input is not floating.
    '''
    # x_train is a numpy object, not a tensor
    if not tf.as_dtype(x_train.dtype).is_floating:
        remarks = f'Input type is {x_train.dtype}'
        return umlaut.errors.InputNotFloatingError(epoch, remarks)


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
            remark = f'During epoch {epoch}, training loss changed by {d_loss:.2f} while validation loss changed by {d_val_loss:.2f}.'
            return umlaut.errors.OverfittingError(epoch, remark)


def check_high_validation_acc(epoch, model, logs):
    val_acc = logs[_get_acc_key(logs, val=True)]
    train_acc = logs[_get_acc_key(logs)]
    remark = ''
    if val_acc > 0.95:
        remark += f'Validation acuracy is very high ({100. * val_acc:.2f}%).\n'
    if val_acc > train_acc:
        remark += f'Validation accuracy ({100 * val_acc:.2f}%) is higher than train accuracy ({100. * train_acc:.2f}%).'
    if remark:
        return umlaut.errors.OverconfidentValAccuracy(epoch, remark)


def check_initialization(epoch, model, logs):
    NotImplemented
