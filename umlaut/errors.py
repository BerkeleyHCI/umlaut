

class BaseErrorMessage:
    title = 'Base Error Message'
    description = 'Base error message. Override me. You shouldn\'t be seeing this in the frontend.'

    def __init__(self, epoch):
        self.epoch = epoch
        self.annotations = [epoch - 1, epoch]

    def __str__(self):
        return '\n'.join((self.title, self.description))

    def __repr__(self):
        return f'<ERROR: {self.title}>'


class InputNotNormalizedError(BaseErrorMessage):
    id_str = 'input_normalization'
    title = 'Input Data Exceeds Typical Limits'
    #TODO really need to handle the formatting in a more clever way
    #  could probably live on the server side and trigger the errors there.
    description = 'Your input data does not look normalized.{}' \
                  '\n#### Solution:  \nYou should normalize the input data where its values fall between the typical ranges of 0 to 1 or -1 to 1, before passing them into the model. E.g., for image data (pixels ranging from 0-255), a typical way to normalize the pixel values to the range of 0 to 1 is:  ' \
                  '\n\n     `training_images = training_images / 255.0`'
    def __init__(self, epoch, remarks=''):
        self.epoch = epoch
        self.annotations = [epoch - 1, epoch]
        if remarks:
            #TODO make remarks formatting better in the future
            self.description = InputNotNormalizedError.description.format(
                ' (' + remarks + ')',
            )
    

class InputNotFloatingError(BaseErrorMessage):
    id_str = 'input_normalization'
    title = 'Input is not a Float type'
    description = 'Your input is not a floating type.' \
                  '\n#### Solution: \nYour input should be a floating point type (supporting decimals), rather than an integer type. This allows gradients to propogate properly to your neural net\'s weights.' \
                  '\nYou can either implicitly change the type of your input (e.g., by dividing by a float): `X_train = X_train / 1.0`, or by setting the `dtype` of your input to something such as `tf.float32`.'


class NaNInLossError(BaseErrorMessage):
    id_str = 'nan-loss0'
    title = 'NaN (Not a number) in loss'
    description = 'The loss value of your model has gone to NaN (could indicate infinity). This could be caused by a learning rate that is too high.' \
                   '\n#### Solution:  \n\nYou can set your learning rate when you create your optimizer object. Typical learning rates for the Adam optimizer are between 0.00001 and 0.01.' \
                   '\nFor example, `model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))`'


class NoSoftmaxActivationError(BaseErrorMessage):
    id_str = 'no-softmax'
    title = 'Loss function expects normalized input'
    description = 'The loss function of your model expects a probability distribution as input (i.e., the likelihood for all the classes sums to 1), but your model is producing un-normalized outputs, called "logits". Logits can be normalized to a probability distribution with a [softmax](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Softmax) layer.' \
                  '\n#### Solution: \n\nMany Keras loss function [classes](https://www.tensorflow.org/api_docs/python/tf/keras/losses) can automatically compute softmax for you by passing in a `from_logits` flag:' \
                  '\n\n`tf.keras.losses.<your loss function here>(from_logits=True)`' \
                  '\n\n where specifying `from_logits=True` will tell keras to apply softmax to your model output before calculating the loss function.' \
                  '\nAlternatively, you can manually add a softmax layer to the end of your model using `tf.keras.Softmax()`.'
    def __init__(self, epoch):
        super().__init__(epoch)
        self.annotations = None  # static check, no annotations