#
# Based off project at
# https://github.com/kenoma/KerasFuzzy
#
# Implemented per description in
# Leng, Prasad, McGinnity (2004)
#
# Andrew Edmonds - 2019
# github.com/andrewre23
#

from keras import backend as K
from keras.engine.topology import Layer


class FuzzyLayer(Layer):
    """
    Class for Fuzzy Layer (2) of sofenn

    -Radial (Ellipsoidal) Basis Function Layer
    -each neuron represents "if-part" or premise
    of a fuzzy rule
    -output is product of Membership Functions (MF)
    -each MF is Gaussian function:
        mu(i,j) = exp{- [x(i) - c(i,j)]^2 / [2 * sigma(i,j)^2]}
        for i features and  j neurons

        mu(i,j)    = ith MF of jth neuron
        c(i,j)     = center of ith MF of jth neuron
        sigma(i,j) = width of ith MF of jth neuron

    -output for fuzzy layer is:
        phi(j) = exp{-sum[i=1,r;
                    [x(i) - c(i,j)]^2 / [2 * sigma(i,j)^2]]}
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
        super(FuzzyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Build objects for processing steps

        Parameters
        ==========
        input_shape : tuple
            - input shape of training data
            - last index will be taken for sizing variables

        Attributes
        ==========
        c : center
            - c(i,j)
            - center of ith membership function of jth neuron

        s : sigma
            - s(i,j)
            - sigma of ith membership function of jth neuron
        """
        self.c = self.add_weight(name='c',
                                 shape=(input_shape[-1], self.output_dim),
                                 initializer=
                                 self.initializer_centers if self.initializer_centers is not None
                                 else 'uniform',
                                 trainable=True)
        self.s = self.add_weight(name='s',
                                 shape=(input_shape[-1], self.output_dim),
                                 initializer=
                                 self.initializer_sigmas if self.initializer_sigmas is not None
                                 else 'ones',
                                 trainable=True)
        super(FuzzyLayer, self).build(input_shape)

    def call(self, x):
        """
        Build processing logic for layer

        Parameters
        ==========
        x : tensor
            - input tensor
            - shape: (features,)

        Attributes
        ==========
        aligned_x : tensor
            - x(i,j)
            - ith feature of jth neuron

        aligned_c : tensor
            - c(i,j)
            - center of ith membership function of jth neuron

        aligned_s : tensor
            - s(i,j)
            - sigma of ith membership function of jth neuron

        Returns
        =======
        phi: tensor
            - phi(neurons,)
            - output of fuzzy layer
        """
        aligned_x = K.repeat_elements(K.expand_dims(x, axis=-1), self.output_dim, -1)
        aligned_c = self.c
        aligned_s = self.s

        phi = K.exp(-K.sum(K.square(aligned_x - aligned_c) / (2 * K.square(aligned_s)),
                           axis=-2, keepdims=False))
        return phi

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.output_dim,)
