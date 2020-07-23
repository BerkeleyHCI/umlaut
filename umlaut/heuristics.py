import re
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from termcolor import colored

import umlaut.errors

def _print_warning(message):
    print(colored('WARNING: ', 'red'), colored(message, 'yellow'))

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
    errors_raised.append(check_missing_activations(model))
    return errors_raised


def run_epoch_heuristics(epoch, model, logs, model_input, source_module):
    errors_raised = []
    errors_raised.append(check_input_shape(epoch, model_input))
    errors_raised.append(check_input_normalization(epoch, model_input, source_module))
    errors_raised.append(check_input_is_floating(epoch, model, model_input, source_module))
    errors_raised.append(check_nan_in_loss(epoch, model, model_input, logs))
    errors_raised.append(check_learning_rate_range(epoch, model))
    errors_raised.append(check_overfitting(epoch, model, logs))
    errors_raised.append(check_high_validation_acc(epoch, logs))
    return errors_raised


def check_accuracy_is_added_to_metrics(logs, source_module):
    NotImplemented


def check_validation_is_added_to_fit(logs, source_module):
    NotImplemented


def check_input_shape(epoch, x_train):
    #TODO x_train is actually model_input (from last batch)
    if x_train is None:
        _print_warning('train data not provided to umlaut, skipping heuristics')
        return
    if K.image_data_format() == 'channels_first':
        if len(x_train.shape) == 4 and x_train.shape[2] != x_train.shape[3]:
            remark = f'Epoch {epoch}: Input shape is not H,C,H,W. Instead got {x_train.shape}'
            return umlaut.errors.InputWrongShapeError(epoch, remark)
    elif len(x_train.shape) == 4 and x_train.shape[1] != x_train.shape[2]:
        remark = f'Epoch {epoch}: Input shape is not N,H,W,C. Instead got {x_train.shape}'
        return umlaut.errors.InputWrongShapeError(epoch, remark)



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
    #TODO x_train is actually model_input (from last batch)
    # x_train is a numpy object, not a tensor
    if x_train is None:
        _print_warning('train data not provided to umlaut, skipping heuristics')
        return
    if not tf.as_dtype(x_train.dtype).is_floating:
        remarks = f'Epoch {epoch}: Input type is {x_train.dtype}'
        module_ref = _get_module_ref_from_pattern('model\.fit', source_module)
        return umlaut.errors.InputNotFloatingError(epoch, remarks, module_ref)


def check_nan_in_loss(epoch, model, x_train, logs):
    '''Returns a NanInLossError if loss is NaN.
    '''
    #TODO x_train is actually model_input (from last batch)
    loss = logs['loss']
    if np.isnan(loss):
        if x_train is None:
            _print_warning('train data not provided to umlaut, skipping heuristics')
        elif any(np.isnan(x_train)):
            return umlaut.errors.NaNInInputError(epoch)


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
    lr = K.eval(model.optimizer.lr)
    if lr > 0.01 or lr < 1e-7:
        remarks = f'Epoch {epoch}: Learning Rate is {lr}'
        if lr > 0.01:
            return umlaut.errors.LRHighError(epoch, remarks)
        else:
            return umlaut.errors.LRLowError(epoch, remarks)


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


def check_high_validation_acc(epoch, logs):
    if epoch < 3:
        # validation accuracy can be a bit random at first, ignore the noise
        return
    val_acc = logs[_get_acc_key(logs, val=True)]
    train_acc = logs[_get_acc_key(logs)]
    remark = ''
    if val_acc > 0.95:
        remark += f'Epoch {epoch}: validation acuracy is very high ({100. * val_acc:.2f}%).\n'
    if val_acc > train_acc:
        remark += f'Epoch {epoch}: validation accuracy ({100 * val_acc:.2f}%) is higher than train accuracy ({100. * train_acc:.2f}%).'
    if remark:
        return umlaut.errors.OverconfidentValAccuracy(epoch, remark)


def check_missing_activations(model):
    err_layers = []
    for i, layer in enumerate(model.layers[:-1]):
        layer_config = layer.get_config()
        if 'activation' in layer_config:
            if layer_config['activation'] == 'linear':
                if i == len(model.layers) - 2 and model.layers[-1].name == 'softmax':
                    continue
                err_layers.append((i, layer.name))
    if err_layers:
        remarks = '\n'.join([f'Layer {l[0]} ({l[1]}) has a missing or linear activation' for l in err_layers])
        return umlaut.errors.MissingActivationError(remarks=remarks)
