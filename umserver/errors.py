import dash_core_components as dcc
import dash_html_components as html


class BaseErrorMessage:
    title = 'Base Error Message'
    subtitle = 'Override me.'
    _md_solution = ['Base error message. Override me.']

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

    def render(self, id_in):
        error_fmt = [
            html.H3(self.title),
            dcc.Markdown(self.subtitle),
            html.H4('Solution'),
            dcc.Markdown(self.description),
        ]
        if self.epochs is None:
            error_fmt.append(html.Small('Captured before start of training.'))
        else:
            error_fmt.append(html.Small(f'Captured at epochs {self.epochs}.'))

        error_fmt.append(html.Button([
            html.Img(
                src='https://cdn.sstatic.net/Sites/stackoverflow/company/Img/logos/so/so-icon.svg',
                height='15px',
                style={'paddingRight': '5px;'},
            ),
            html.Small('Search Stack Overflow'),
        ]))
        error_fmt.append(html.Hr())

        return html.Div(
            error_fmt,
            id=id_in,
            style={'cursor': 'pointer', 'display': 'inline-block'},
        )

    def __init__(self, epochs, *args, **kwargs):
        self.epochs = epochs
        if type(self.epochs) is not list:
            self.epochs = [epochs]

    def __str__(self):
        return '\n'.join((self.title, self.description))

    def __repr__(self):
        return f'<ERROR: {self.title}>'


class InputNotNormalizedError(BaseErrorMessage):
    title = 'Input Data Exceeds Typical Limits'
    subtitle = 'Your input data does not look normalized.{}'
    _md_solution = [
        'You should normalize the input data where its values fall between the typical ranges of 0 to 1 or -1 to 1, before passing them into the model. E.g., for image data (pixels ranging from 0-255), a typical way to normalize the pixel values to the range of -1 to 1 is:',
        '`your_input_images = (your_input_images / 127.0) - 1`',
    ]
    def __init__(self, epochs, remarks=''):
        super().__init__(epochs)
        self.subtitle = InputNotNormalizedError.subtitle.format(
            ' (' + remarks + ')',
        )
    

class InputNotFloatingError(BaseErrorMessage):
    title = 'Input is not a Float type'
    subtitle = 'Your input is not a floating type.'
    _md_description = [
        'Your input should be a floating point type (supporting decimals), rather than an integer type. This allows gradients to propogate properly to your neural net\'s weights.',
        'You can either implicitly change the type of your input (e.g., by dividing by a float): `X_train = X_train / 1.0`, or by setting the `dtype` of your input to something such as `tf.float32`.',
    ]


class NaNInLossError(BaseErrorMessage):
    title = 'NaN (Not a number) in loss'
    subtitle = 'The loss value of your model has gone to NaN (could indicate infinity). This could be caused by a learning rate that is too high.'
    _md_solution = [
        'You can set your learning rate when you create your optimizer object. Typical learning rates for the Adam optimizer are between 0.00001 and 0.01. For example:',
        '`model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))`',
    ]


class NoSoftmaxActivationError(BaseErrorMessage):
    title = 'Loss function expects normalized input'
    subtitle = 'The loss function of your model expects a probability distribution as input (i.e., the likelihood for all the classes sums to 1), but your model is producing un-normalized outputs, called "logits". Logits can be normalized to a probability distribution with a [softmax](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Softmax) layer.'
    _md_solution = [
        'Many Keras loss function [classes](https://www.tensorflow.org/api_docs/python/tf/keras/losses) can automatically compute softmax for you by passing in a `from_logits` flag:',
        '`tf.keras.losses.<your loss function here>(from_logits=True)`',
        'where specifying `from_logits=True` will tell keras to apply softmax to your model output before calculating the loss function.',
        'Alternatively, you can manually add a softmax layer to the end of your model using `tf.keras.Softmax()`.',
    ]

    def get_annotations(self):
        return None  # static check, no annotations

    def __init__(self, *args, **kwargs):
        self.epochs = None


class OverfittingError(BaseErrorMessage):
    title = 'Possible Overfitting'
    subtitle = 'The validation loss is increasing while training loss is stuck or decreasing. This could indicate overfitting.'
    _md_solution = [
        'Try reducing the power of your model or adding regularlization. You can reduce the power of your model by decreasing the `units` or `filters` parameters of `Dense` or `Conv2D` layers.',
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