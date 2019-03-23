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


class WeightedLayer(Layer):
    """
    Weighted Layer (4) of SOFNN
    ===========================

    - Weighting of ith MF of each feature

    - yields the "consequence" of the jth fuzzy rule of fuzzy model
    - each neuron has two inputs:
        - output of previous related neuron j
        - weighted bias w2j
    - with:
        r      = number of original input features

        B      = [1, x1, x2, ... xr]
        Aj     = [aj0, aj1, ... ajr]

        w2j    = Aj * B =
                 aj0 + aj1x1 + aj2x2 + ... ajrxr

        PHI(j) = output of jth neuron from
                normalized layer

    -output for fuzzy layer is:
        fj     = w2j PHI(j)
    """

    def __init__(self,
                 output_dim,
                 initializer_a=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.output_dim = output_dim
        self.initializer_a = initializer_a
        super(WeightedLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Build objects for processing steps

        Parameters
        ==========
        input_shape : list of tuples
            - [x shape, phi shape]
            - x shape: (features,)
            - phi shape: (neurons,)

        Attributes
        ==========
        a : then-part (consequence) of fuzzy rule
            - a(i,j)
            - trainable weight of ith feature of jth neuron
            - shape: (1+features, neurons)
        """
        assert isinstance(input_shape, list)
        self.a = self.add_weight(name='a',
                                 shape=(1 + input_shape[0][-1], self.output_dim),
                                 initializer=self.initializer_a if
                                 self.initializer_a is not None else 'uniform',
                                 trainable=True)
        super(WeightedLayer, self).build(input_shape)

    def call(self, x):
        """
        Build processing logic for layer

        Parameters
        ==========
        x : list of tensors
            - list of tensor with input data and phi output of previous layer
            - [x, phi]
            - x shape: (features,)
            - phi shape: (neurons,)

        Attributes
        ==========
        aligned_b : tensor
            - input vector with [1.0] prepended for bias weight
            - shape: (1+features,)

        aligned_c : tensor
            - c(i,j)
            - center of ith membership function of jth neuron
            - shape: (features, neurons)

        aligned_s : tensor
            - s(i,j)
            - sigma of ith membership function of jth neuron
            - shape: (features, neurons)

        Returns
        =======
        f: tensor
            - phi(neurons,)
            - output of each neuron in fuzzy layer
            - shape: (neurons,)
        """
        assert isinstance(x, list)
        x, phi = x

        # align tensors by prepending bias value for input tensor in b
        aligned_b = K.concatenate([K.tf.convert_to_tensor([1], dtype=x.dtype), x])
        aligned_a = K.transpose(self.a)

        return phi * K.sum(aligned_a * aligned_b, axis=-1)

    def compute_output_shape(self, input_shape):
        """
        Return output shape of input data

        Parameters
        ==========
        input_shape : tuple
            - shape of input data
            - shape: (samples, features)

        Returns
        =======
        output_shape : tensor
            - output shape of layer
            - shape: (samples, neurons)
        """
        assert isinstance(input_shape, list)
        x_shape, phi_shape = input_shape
        return tuple(x_shape[:-1]) + (self.output_dim,)
