
class InputNotNormalizedError:
    title = 'Input Data Exceeds Typical Limits'
    #TODO really need to handle the formatting in a more clever way
    #  could probably live on the server side and trigger the errors there.
    description = 'Your input data does not look normalized.{}' \
                  '\n#### Solution:  \nYou should normalize the input data (setting the range X_train and X_test to be from -1 to 1) before passing them into the model. For image data (pixels ranging from 0-255), a typical way to normalize the pixel values is:  ' \
                  '\n\n     `X_train = X_train / 128.0 - 1`'
    id_str = 'input_normalization'
    def __init__(self, epoch, remarks=''):
        self.epoch = epoch
        self.annotations = [epoch - 1, epoch]
        if remarks:
            #TODO make remarks formatting better in the future
            self.description = InputNotNormalizedError.description.format(
                ' (' + remarks + ')',
            )
    
    def __str__(self):
        return '\n'.join((self.title, self.description))

    def __repr__(self):
        return f'<ERROR: {self.title}>'