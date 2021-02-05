import dash_core_components as dcc
import dash_html_components as html
from urllib import parse


REMARKS_STYLE = {
    # format to look like a slack inline code block, except pink
    'backgroundColor': 'rgba(29, 28, 29, 0.04)',
    'color': 'rgb(224, 30, 90)',
    'border': '1px solid rgb(210, 210, 210)',
    'borderRadius': '4px',
    'padding': '8px',
    'width': '100%',
    'overflow': 'scroll',
    'display': 'inline-block',
    'maxHeight': '2.5rem',
    'box-shadow': 'inset 0px 0px 3px 0px #ccc',
    'marginBottom': '-4px',
}


def get_error_color(error_idx):
    '''Make a qualitative color range for 4 colors (90 degrees)'''
    return f'hsl({(25 + 90*error_idx) % 360}, 95%, 80%)'


class BaseErrorMessage:
    title = 'Base Error Message'
    subtitle = 'Base subtitle message'
    _md_solution = None
    _so_query = None
    _docs_url = None

    @property
    def description(self):
        return '\n\n'.join(
            self._md_solution,
        )

    @property
    def is_static(self):
        return self.epochs is None

    @staticmethod
    def _render_icon(img_url, caption, href):
        return html.A(
            [
                html.Img(
                    src=img_url,
                    height='15px',
                    style={'paddingTop': '1.2rem', 'paddingRight': '5px', 'margin-bottom': '-3px'},
                ),
                caption,
            ],
            href=href,
            target='_blank',
            style={'paddingRight': '1rem'},
        )

    def get_annotations(self):
        return [(e - 1, e) for e in self.epochs]

    def serialized(self):
        return self.__dict__

    def render(self, error_index):
        '''Formats an error message as a Dash html component.
        
        This method assigns the id "types" of 'error-msg' and
        'error-msg-btn-annotate' which are used by callbacks.
        '''
        error_fmt = [
            html.Span(
                [
                    html.Span(id={'type': 'error-msg-indicator', 'index': error_index}, style={
                        'backgroundColor': get_error_color(error_index),
                        'borderRadius': '50%',
                        'marginRight': '5px',
                        'display': 'inline-block',
                    }),
                    html.H3(self.title, style={'display': 'inline-block'}),
                ],
                style={'cursor': 'pointer'},
            ),
            dcc.Markdown(self.subtitle),
        ]

        # add error context as a formatted <pre>
        if self.remarks:
            error_fmt.append(html.Pre(
                self.remarks,
                style=REMARKS_STYLE,
            ))

        error_fmt.extend([
            html.H4('Solution'),
            dcc.Markdown(self.description),
        ])

        # write where error was captured
        if self.epochs is None:
            error_fmt.append(html.Small('Captured before start of training.'))
        else:
            error_fmt.append(html.Small(f'Captured at epochs {self.epochs}.'))

        error_fmt.append(html.Br())

        # append icons + external refs to error
        if self._so_query:
            error_fmt.append(self._render_icon(
                img_url='https://cdn.sstatic.net/Sites/stackoverflow/company/Img/logos/so/so-icon.svg',
                caption='Search Stack Overflow',
                href=f'https://stackoverflow.com/search?{parse.urlencode(self._so_query)}',
            ))
        if self._docs_url:
            error_fmt.append(self._render_icon(
                img_url='https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg',
                caption='Search Docs',
                href=self._docs_url,
            ))
        if self.module_url:
            error_fmt.append(self._render_icon(
                img_url='https://upload.wikimedia.org/wikipedia/commons/9/9a/Visual_Studio_Code_1.35_icon.svg',
                caption='Open in VSCode',
                href=f'vscode://file{self.module_url}',
            ))
        error_fmt.append(html.Hr())

        return html.Div(
            error_fmt,
            id={'type': 'error-msg', 'index': error_index},
            style={'display': 'inline-block'},
        )

    def __init__(
        self,
        epochs,
        remarks=None,
        module_url=None,
        *args,
        **kwargs,
    ):
        self.epochs = epochs
        self.remarks = remarks
        self.module_url = module_url
        if epochs is not None and type(self.epochs) is not list:
            self.epochs = [epochs]

    def __str__(self):
        return '\n'.join((self.title, self.description))

    def __repr__(self):
        return f'<{self.title}>'


class InputNotNormalizedError(BaseErrorMessage):
    title = 'Error: Input data exceeds typical limits'
    subtitle = 'Your input data does not look normalized.'
    _so_query = {'q': '[keras] closed:yes normalization'}
    _docs_url = 'https://www.tensorflow.org/tutorials/keras/classification#preprocess_the_data'
    _md_solution = [
        'You should normalize the input data so its values fall between the typical ranges of -1 to 1 before passing them into the model',
        'For image data, (pixels ranging from 0-255), a typical way to normalize the pixel values to the range of -1 to 1 is',
        '`training_images = (training_images / 128.0) - 1`',
    ]
    

class InputWrongShapeError(BaseErrorMessage):
    title = 'Error: Image data may have incorrect shape'
    subtitle = 'Input image data '
    _md_description = [
        'Your input is 4-dimensional with 2 equal dimensions, which is typically an image type. Most keras layers by default expect image data to be formatted as "NHWC" (Batch_size, Height, Width, Channel) unless otherwise specified. If running on CPU, setting the Keras image backend to \'channels_first\' and using "NCHW" (Batch_size, Channel, Height, Width) may sometimes improve performance.',
        'You can transpose your input data to move your channels last using `tf.transpose(X_train_images, [0, 2, 3, 1])`.',
    ]


class InputNotFloatingError(BaseErrorMessage):
    title = 'Error: Input is not a Float type'
    subtitle = 'Your input is not a floating type.'
    _so_query = {'q': '[keras] closed:yes float'}
    _docs_url = 'https://www.tensorflow.org/tutorials/keras/classification#preprocess_the_data'
    _md_description = [
        'Your input should be a floating point type (supporting decimals), rather than an integer type. This allows gradients to propogate properly to your neural net\'s weights.',
        'You can either implicitly change the type of your input (e.g., by dividing by a float): `X_train = X_train / 1.0`, or by setting the `dtype` of your input to something such as `tf.float32`.',
    ]


class LRError(BaseErrorMessage):
    _md_solution = [
        'You can set your learning rate when you create your optimizer object. Typical learning rates for the Adam optimizer are between 0.00001 and 0.01. For example:',
        '`model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))`',
    ]
    _docs_url = 'https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam'


class LRHighError(LRError):
    title = 'Warning: Learning Rate is high'
    subtitle = 'The learning rate you set is higher than the typical range. This could lead to the model\'s inability to learn. This can also lead to NaN loss values.'
    

class LRLowError(LRError):
    title = 'Warning: Learning Rate is low'
    subtitle = 'The learning rate you set is lower than the typical range. This could lead to the model\'s inability to learn. This can also lead to NaN loss values.'
    

class NaNInInputError(BaseErrorMessage):
    title = 'Critical: NaN (Not a number) in input'
    subtitle = 'Some values in your model input is NaN (could indicate infinity).'
    _so_query = {'q': '[keras] nan input'}
    _md_solution = [
        'Please double check your input and make sure no NaN exists in it.',
    ]


class MissingActivationError(BaseErrorMessage):
    title = 'Critical: Missing activation functions'
    subtitle = 'The model has layers without nonlinear activation functions. This may limit the model\'s ability to learn since stacked `Dense` layers without activations will mathematically collapse to a single `Dense` layer.'
    _md_solution = [
        'Make sure the `activation` argument is passed into your `Dense` and Convolutional (e.g., `Conv2D`) layers.',
        'A common practice is to use `activation=\'relu\'`.'
    ]
    _docs_url = 'https://www.tensorflow.org/api_docs/python/tf/keras/activations'

    def get_annotations(self):
        return None  # static check, no annotations

    def __init__(self, epochs, remarks=None, module_url=None, *args, **kwargs):
        # set epochs to None
        self.epochs = None
        self.remarks = remarks
        self.module_url = module_url



class NoSoftmaxActivationError(BaseErrorMessage):
    title = 'Critical: Missing Softmax layer before loss'
    subtitle = 'The loss function of your model expects a probability distribution as input (i.e., the likelihood for all the classes sums to 1), but your model is producing un-normalized outputs, called "logits". Logits can be normalized to a probability distribution with a [softmax](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Softmax) layer.'
    _so_query = {'q': '[keras] is:closed from_logits'}
    _docs_url = 'https://www.tensorflow.org/api_docs/python/tf/keras/losses'
    _md_solution = [
        'Many Keras loss function [classes](https://www.tensorflow.org/api_docs/python/tf/keras/losses) can automatically compute softmax for you by passing in a `from_logits` flag:',
        '`tf.keras.losses.<your loss function class here>(from_logits=True)`',
        'where specifying `from_logits=True` will tell keras to apply softmax to your model output before calculating the loss function.',
        'Alternatively, you can manually add a softmax layer to the end of your model using `tf.keras.layers.Softmax()`.',
    ]

    def get_annotations(self):
        return None  # static check, no annotations

    def __init__(self, epochs, remarks=None, module_url=None, *args, **kwargs):
        # set epochs to None
        self.epochs = None
        self.remarks = remarks
        self.module_url = module_url


class OverconfidentValAccuracy(BaseErrorMessage):
    title = 'Warning: Check validation accuracy'
    subtitle = 'The validation accuracy is either higher than typical results (near 100%) or higher than training accuracy (which can suggest problems with data labeling or splitting). However, during early epochs, this could be a false positive.'
    _so_query = {'q': '[keras] validation accuracy high'}
    _md_solution = [
        'A high validation accuracy (around 100%) can indicate a problem with data labels, overlap between the training and validation data, or differences in preparing data for training and evaluation.',
        'Check to see how the model performs on the test set (data the model has not seen before). If the test accuracy is similarly high, inspect the predictions by hand and ensure they make sense.',
    ]


class OverfittingError(BaseErrorMessage):
    title = 'Warning: Possible overfitting'
    subtitle = 'The validation loss is increasing while training loss is stuck or decreasing. This could indicate overfitting. However, if validation loss is still trending downwards afterwards, this error could be a false positive.'
    _so_query = {'q': '[keras] is:closed regularization'}
    _docs_url = 'https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/Regularizer'
    _md_solution = [
        'Try adding dropout or reducing the power of your model.',
        'Dropout randomly omits weight updates during training (with some probability) which decreases model power and potentially increases robustness.'
        'You can reduce the power of your model by decreasing the `units` or `filters` parameters of `Dense` or `Conv2D` layers.',
    ]


class FinalLayerHasActivationError(BaseErrorMessage):
    title = 'Warning: Last model layer has nonlinear activation'
    subtitle = 'The last layer of the model has a nonlinear activation function before Softmax. This can clip gradient updates and prevent the model from learning.'
    _md_solution = [
        'Remove the `activation` argument from the last layer of your model.'
    ]

    def get_annotations(self):
        return None  # static check, no annotations

    def __init__(self, epochs, remarks=None, module_url=None, *args, **kwargs):
        # set epochs to None
        self.epochs = None
        self.remarks = remarks
        self.module_url = module_url


class HighDropoutError(BaseErrorMessage):
    title = 'Warning: High dropout rate'
    subtitle = 'The dropout parameter of the indicated layer(s) is above 0.5, meaning less than half of the gradient updates will propagate through. This can prevent your model from learning.'
    _md_solution = [
        'Lower the dropout rate. Typical values range between \[0.2, 0.3\], extending to \[0.1, 0.5\].',
    ]

    def get_annotations(self):
        return None  # static check, no annotations

    def __init__(self, epochs, remarks=None, module_url=None, *args, **kwargs):
        # set epochs to None
        self.epochs = None
        self.remarks = remarks
        self.module_url = module_url


ERROR_KEYS = {
    'input_normalization': InputNotNormalizedError,
    'input_not_floating': InputNotFloatingError,
    'input_wrong_shape': InputWrongShapeError,
    'nan_input': NaNInInputError,
    'lr_high': LRHighError,
    'lr_low': LRLowError,
    'no_softmax': NoSoftmaxActivationError,
    'overfitting': OverfittingError,
    'overconfident_val': OverconfidentValAccuracy,
    'missing_activations': MissingActivationError,
    'activation_final_layer': FinalLayerHasActivationError,
    'high_dropout_rate': HighDropoutError,
}

# assign id strings to error messages as a backref
for error_id in ERROR_KEYS:
    ERROR_KEYS[error_id].id_str = error_id