#
# SOFENN
# Self-Organizing Fuzzy Neural Network
#
# (sounds like soften)
#
#
# Implemented per description in
# An on-line algorithm for creating self-organizing
# fuzzy neural networks
# Leng, Prasad, McGinnity (2004)
#
#
# Andrew Edmonds - 2019
# github.com/andrewre23
#


from keras import backend as K
from keras.engine.topology import Layer


class OutputLayer(Layer):
    """
    Output Layer (5) of SOFNN
    ==========================

    - Sum of each output of previous layer (f)

    - output for fuzzy layer is:
        sum[k=1, u; f(k)]
                for u neurons
    """

    def __init__(self,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.output_dim = 1
        super(OutputLayer, self).__init__(name='RawOutput', **kwargs)

    def build(self, input_shape):
        """
        Build objects for processing steps

        Parameters
        ==========
        input_shape : tuple
            - f shape : (samples, neurons)
        """
        super(OutputLayer, self).build(input_shape)

    def call(self, x):
        """
        Build processing logic for layer

        Parameters
        ==========
        x : tensor
            - tensor with f as output of previous layer
            - f shape: (samples, neurons)

        Returns
        =======
        output: tensor
            sum[k=1, u; f(k)]
                for u neurons
        - sum of all f's from previous layer
            - shape: (samples,)
        """
        # get raw sum of all neurons for each sample
        sums = K.sum(x, axis=-1)
        return K.repeat_elements(K.expand_dims(sums, axis=-1), self.output_dim, -1)

    def compute_output_shape(self, input_shape):
        """
        Return output shape of input data

        Parameters
        ==========
        input_shape : tuple
            - f shape: (samples, neurons)

        Returns
        =======
        output_shape : tuple
            - output shape of final layer
            - shape: (samples,)
        """
        return tuple(input_shape[:-1]) + (self.output_dim,)
