import dash_core_components as dcc
import dash_html_components as html
from urllib import parse


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
            html.Span(id={'type': 'error-msg-indicator', 'index': error_index}, style={
                'backgroundColor': get_error_color(error_index),
                'borderRadius': '50%',
                'marginRight': '5px',
                'display': 'inline-block',
            }),
            html.H3(self.title, style={'display': 'inline-block'}),
            dcc.Markdown(self.subtitle),
        ]

        if self.remarks:
            error_fmt.append(html.Code(
                self.remarks,
                style={
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
                },
            ))

        error_fmt.extend([
            html.H4('Solution'),
            dcc.Markdown(self.description),
        ])

        if self.epochs is None:
            error_fmt.append(html.Small('Captured before start of training.'))
        else:
            error_fmt.append(html.Small(f'Captured at epochs {self.epochs}.'))

        error_fmt.append(html.Br())
        if self._so_query:
            error_fmt.append(html.A(
                [
                    html.Img(
                        src='https://cdn.sstatic.net/Sites/stackoverflow/company/Img/logos/so/so-icon.svg',
                        height='15px',
                        style={'paddingTop': '1.2rem', 'paddingRight': '5px', 'margin-bottom': '-3px'},
                    ),
                    'Search Stack Overflow',
                ],
                href='https://stackoverflow.com/search?{}'.format(
                    parse.urlencode(self._so_query),
                ),
                target='_blank',
                style={'paddingRight': '1rem'},
            ))
        if self._docs_url:
            error_fmt.append(html.A(
                [
                    html.Img(
                        src='https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg',
                        height='15px',
                        style={'paddingTop': '1.2rem', 'paddingRight': '5px', 'margin-bottom': '-3px'},
                    ),
                    'Search Docs',
                ],
                href=self._docs_url,
                target='_blank',
                style={'paddingRight': '1rem'},
            ))
        if self.module_url:
            error_fmt.append(html.A(
                [
                    html.Img(
                        src='https://upload.wikimedia.org/wikipedia/commons/9/9a/Visual_Studio_Code_1.35_icon.svg',
                        height='15px',
                        style={'paddingTop': '1.2rem', 'paddingRight': '5px', 'margin-bottom': '-3px'},
                    ),
                    'Open in VSCode',
                ],
                href=f'vscode://file{self.module_url}',
            ))
        error_fmt.append(html.Hr())

        return html.Div(
            error_fmt,
            id={'type': 'error-msg', 'index': error_index},
            style={'cursor': 'pointer', 'display': 'inline-block'},
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
        return f'<ERROR: {self.title}>'


class InputNotNormalizedError(BaseErrorMessage):
    title = 'Input data exceeds typical limits'
    subtitle = 'Your input data does not look normalized.'
    _so_query = {'q': '[keras] closed:yes normalization'}
    _docs_url = 'https://www.tensorflow.org/tutorials/keras/classification#preprocess_the_data'
    _md_solution = [
        'You should normalize the input data so its values fall between the typical ranges of -1 to 1 before passing them into the model',
        'For image data, (pixels ranging from 0-255), a typical way to normalize the pixel values to the range of -1 to 1 is',
        '`training_images = (training_images / 128.0) - 1`',
    ]
    

class InputNotFloatingError(BaseErrorMessage):
    title = 'Input is not a Float type'
    subtitle = 'Your input is not a floating type.'
    _so_query = {'q': '[keras] closed:yes float'}
    _docs_url = 'https://www.tensorflow.org/tutorials/keras/classification#preprocess_the_data'
    _md_description = [
        'Your input should be a floating point type (supporting decimals), rather than an integer type. This allows gradients to propogate properly to your neural net\'s weights.',
        'You can either implicitly change the type of your input (e.g., by dividing by a float): `X_train = X_train / 1.0`, or by setting the `dtype` of your input to something such as `tf.float32`.',
    ]


class NaNInLossError(BaseErrorMessage):
    title = 'NaN (Not a number) in loss'
    subtitle = 'The loss value of your model has gone to NaN (could indicate infinity). This could be caused by a learning rate that is too high.'
    _so_query = {'q': '[keras] nan loss'}
    _md_solution = [
        'You can set your learning rate when you create your optimizer object. Typical learning rates for the Adam optimizer are between 0.00001 and 0.01. For example:',
        '`model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))`',
    ]


class NoSoftmaxActivationError(BaseErrorMessage):
    title = 'Loss function expects normalized input'
    subtitle = 'The loss function of your model expects a probability distribution as input (i.e., the likelihood for all the classes sums to 1), but your model is producing un-normalized outputs, called "logits". Logits can be normalized to a probability distribution with a [softmax](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Softmax) layer.'
    _so_query = {'q': '[keras] is:closed from_logits'}
    _docs_url = 'https://www.tensorflow.org/api_docs/python/tf/keras/losses'
    _md_solution = [
        'Many Keras loss function [classes](https://www.tensorflow.org/api_docs/python/tf/keras/losses) can automatically compute softmax for you by passing in a `from_logits` flag:',
        '`tf.keras.losses.<your loss function here>(from_logits=True)`',
        'where specifying `from_logits=True` will tell keras to apply softmax to your model output before calculating the loss function.',
        'Alternatively, you can manually add a softmax layer to the end of your model using `tf.keras.Softmax()`.',
    ]

    def get_annotations(self):
        return None  # static check, no annotations

    def __init__(self, remarks=None, *args, **kwargs):
        # set epochs to None
        self.epochs = None
        self.remarks = remarks


class OverfittingError(BaseErrorMessage):
    title = 'Possible Overfitting'
    subtitle = 'The validation loss is increasing while training loss is stuck or decreasing. This could indicate overfitting.'
    _so_query = {'q': '[keras] is:closed regularization'}
    _docs_url = 'https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/Regularizer'
    _md_solution = [
        'Try reducing the power of your model or adding regularization. You can reduce the power of your model by decreasing the `units` or `filters` parameters of `Dense` or `Conv2D` layers.',
        'Regularization penalizes weights which are high in magnitude. You can try adding L2 or L1 regularization by using a [regularizer](https://www.tensorflow.org/api_docs/python/tf/keras/regularizers).'
    ]

ERROR_KEYS = {
    'input_normalization': InputNotNormalizedError,
    'input_not_floating': InputNotFloatingError,
    'nan_loss': NaNInLossError,
    'no_softmax': NoSoftmaxActivationError,
    'overfitting': OverfittingError,
}

# assign id strings to error messages as a backref
for error_id in ERROR_KEYS:
    ERROR_KEYS[error_id].id_str = error_id