#
# Class for Self-Organizing Fuzzy Neural Network
#
# Implemented per description in
# An on-line algorithm for creating self-organizing
# fuzzy neural networks
# Leng, Prasad, McGinnity (2004)
#
#
# Andrew Edmonds - 2019
#

import numpy as np
import pandas as pd

from keras import backend as K
from keras.models import Model
from keras import regularizers
from keras.layers import Input, LSTM, Dense, Dropout

# custom Fuzzy Layers
from notebooks.modeling.keras.layers import \
    FuzzyLayer, NormalizedLayer, WeightedLayer


class SOFNN(Model):
    """
    Class for building Self-Organizing Fuzzy Neural Networks

    -Composed of 5 layers with varying "fuzzy rule" nodes


    Layers
    ======
    1 - Input Layer
            input dataset
        - input shape  : (*, features)
    2 - Radial Basis Function Layer (Fuzzy Layer)
            layer to hold fuzzy rules for complex system
        - input shape  : (*, features * neurons)
        - output shape : (*, neurons)
    3 - Normalized Layer
            normalize each output of previous layer as
            relative amount from sum of all previous outputs
        - input shape  : (*, neurons)
        - output shape : (*, neurons)
    4 - Weighted Layer
            multiply bias vector (1+n_features, neurons) by
            parameter vector (1+n_features,) of parameters
            from each fuzzy rule
            multiply each product by output of each rule's
            layer from normalized layer
        - input shape  : (1+features, 1) <- Aj
        - input shape  : (1+features, 1) <- B
        - output shape : (*, neurons)
    5 - Output Layer
            summation of incoming signals from weighted layer
        - input shape  : (*, neurons)
        - output shape : (*,)

    * = samples
    """

    def __init__(self, **kwargs):
        # initialize as Model object
        super().__init__(**kwargs)

        # initialize variables needed for processing
        pass
