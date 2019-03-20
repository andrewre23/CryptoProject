# implemented per description in
# Leng, Prasad, McGinnity (2004)


from keras import backend as K
from keras.engine.topology import Layer


class WeightedLayer(Layer):
    """
    Class for Weighted Layer (4) of SOFNN

    -yields the "consequence" of the jth
        fuzzy rule of fuzzy model
    -each neuron has two inputs:
        -output of previous related neuron j
        -weighted bias w2j
    -with:
        r      = number of original input features

        B      = [1, x1, x2, ... xr].T
        Aj     = [aj0, aj1, ... ajr].T

        w2j    = Aj * B =
                 aj0 + aj1x1 + aj2x2 + ... ajrxr

        PHI(j) = output of jth neuron from
                normalized layer

    -output for fuzzy layer is:
        fj     = w2j PHI(j)
    """
    def __init__(self,
                 output_dim,
                 initializer_centers=None,
                 initializer_sigmas=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.output_dim = output_dim
        self.initializer_centers = initializer_centers
        self.initializer_sigmas = initializer_sigmas
        super(WeightedLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dimensions = list(input_shape)[:-1:-1]
        self.c = self.add_weight(name='c',
                                 shape=(input_shape[-1], self.output_dim),
                                 initializer=self.initializer_centers if self.initializer_centers is not None else 'uniform',
                                 trainable=True)
        self.a = self.add_weight(name='a',
                                 shape=(input_shape[-1], self.output_dim),
                                 initializer=self.initializer_sigmas if self.initializer_sigmas is not None else 'ones',
                                 trainable=True)
        super(FuzzyLayer, self).build(input_shape)

    def call(self, x):

        aligned_x = K.repeat_elements(K.expand_dims(x, axis=-1), self.output_dim, -1)
        aligned_c = self.c
        aligned_a = self.a
        for dim in self.input_dimensions:
            aligned_c = K.repeat_elements(K.expand_dims(aligned_c, 0), dim, 0)
            aligned_a = K.repeat_elements(K.expand_dims(aligned_a, 0), dim, 0)

        xc = K.exp(-K.sum(K.square((aligned_x - aligned_c) / (2 * aligned_a)), axis=-2, keepdims=False))
        # sums = K.sum(xc,axis=-1,keepdims=True)
        # less = K.ones_like(sums) * K.epsilon()
        return xc  # xc / K.maximum(sums, less)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.output_dim,)
