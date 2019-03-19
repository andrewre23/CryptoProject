# based off project at
# https://github.com/kenoma/KerasFuzzy

from keras import backend as K
from keras.layers import Layer


class BaseLayer(Layer):
    """Base template for custom Keras layer"""

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(BaseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(BaseLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class BaseLayerMulti(Layer):
    """Base template for custom Keras layer with multi input and output
        -assume inputs are now lists"""

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(BaseLayerMulti, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(BaseLayerMulti, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        a, b = x
        return [K.dot(a, self.kernel) + b, K.mean(b, axis=-1)]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]]
