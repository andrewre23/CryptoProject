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

import numpy as np
import pandas as pd

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Activation

# custom Fuzzy Layers
from .layers import \
    FuzzyLayer, NormalizedLayer, WeightedLayer


class SOFNN(object):
    """
    Class for building Self-Organizing Fuzzy Neural Network
    =======================================================

    -Implemented per description in:
        "An on-line algorithm for creating self-organizing
        fuzzy neural networks" - Leng, Prasad, McGinnity (2004)
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

    Parameters
    ==========
    - X_train : training input data
        - shape :(samples, features)
    - X_test  : testing input data
        - shape: (samples, features)
    - y_train : training output data
        - shape: (features,)
    - y_test  : testing output data
        - shape: (features,)

    Functions
    =========
    - system_error_check :
        - system error considers generalized performance of overall network
        - add neuron if error above predefined error threshold (delta)
    - if_part_check :
        - if-part criterion checks if current fuzzy rules cover/cluster input vector suitably
        -
    - add_neuron :
        - add one neuron to model
    - prune_neuron :
        - remove neuron from model


    """

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
