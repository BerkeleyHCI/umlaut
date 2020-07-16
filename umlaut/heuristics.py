import re
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


def _search_source_module(pattern, source_module_contents):
    line_matches = []
    for i, line in enumerate(source_module_contents):
        match = re.search(pattern, line)
        if match:
            line_matches.append((i, match))
    return line_matches


def _make_vscode_url(line_match, source_module_path):
    return f'{source_module_path}:{line_match[0]+1}:{line_match[1].start()+1}'


def _get_module_ref_from_pattern(pattern, source_module):
    line_matches = _search_source_module(pattern, source_module['contents'])
    if line_matches:
        return _make_vscode_url(line_matches[0], source_module['path'])
    else:
        return None


def run_pretrain_heuristics(model, source_module):
    errors_raised = []
    errors_raised.append(check_softmax_computed_before_loss(model, source_module))
    return errors_raised


def run_epoch_heuristics(epoch, model, logs, x_train, source_module):
    errors_raised = []
    errors_raised.append(check_input_normalization(epoch, x_train, source_module))
    errors_raised.append(check_input_is_floating(epoch, model, x_train, source_module))
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
        remark = remark + f'Epoch {epoch}: minimum input value is {x_min}, less than the typical value of -1.'
    if x_max > 1:
        remark = remark + f'Epoch {epoch}: maximum input value is {x_max}, greater than the typical value of 1.'
    if remark:
        module_ref = _get_module_ref_from_pattern('model\.fit', source_module)
        return umlaut.errors.InputNotNormalizedError(epoch, remark, module_ref)


def check_input_is_floating(epoch, model, x_train, source_module):
    '''Returns an `InputNotFloatingError` if input is not floating.
    '''
    # x_train is a numpy object, not a tensor
    if not tf.as_dtype(x_train.dtype).is_floating:
        remarks = f'Epoch {epoch}: Input type is {x_train.dtype}'
        module_ref = _get_module_ref_from_pattern('model\.fit', source_module)
        return umlaut.errors.InputNotFloatingError(epoch, remarks, module_ref)


def check_nan_in_loss(epoch, logs):
    '''Returns a NanInLossError if loss is NaN.
    '''
    loss = logs['loss']
    if np.isnan(loss):
        return umlaut.errors.NaNInLossError(epoch)


def check_softmax_computed_before_loss(model, source_module):
    '''Ensures the loss function used has a proper from_logits setting.
    '''
    # from_logits is always False by default per source code
    from_logits = False
    if issubclass(type(model.loss), tf.keras.losses.Loss):
        # get from_logits arg from Loss class family
        from_logits = model.loss._fn_kwargs.get('from_logits', False)
    last_layer_is_softmax = isinstance(model.layers[-1], tf.keras.layers.Softmax)
    if not last_layer_is_softmax and not from_logits:
        line_matches = _search_source_module('model\.add', source_module['contents'])[::-1]
        if not line_matches:
            line_matches = _search_source_module('tf\.keras\.Model', source_module['contents'])
        if not line_matches:
            line_matches = _search_source_module('tf\.keras\.Sequential', source_module['contents'])
        return umlaut.errors.NoSoftmaxActivationError(module_url=_make_vscode_url(line_matches[0], source_module['path']))


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
            remark = f'Epoch {epoch}: training loss changed by {d_loss:.2f} while validation loss changed by {d_val_loss:.2f}.'
            return umlaut.errors.OverfittingError(epoch, remark)


def check_high_validation_acc(epoch, model, logs):
    val_acc = logs[_get_acc_key(logs, val=True)]
    train_acc = logs[_get_acc_key(logs)]
    remark = ''
    if val_acc > 0.95:
        remark += f'Epoch {epoch}: validation acuracy is very high ({100. * val_acc:.2f}%).\n'
    if val_acc > train_acc:
        remark += f'Epoch {epoch}: validation accuracy ({100 * val_acc:.2f}%) is higher than train accuracy ({100. * train_acc:.2f}%).'
    if remark:
        return umlaut.errors.OverconfidentValAccuracy(epoch, remark)


def check_initialization(epoch, model, logs):
    NotImplemented
