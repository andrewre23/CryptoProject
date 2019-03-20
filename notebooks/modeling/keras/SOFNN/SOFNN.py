#
# Class for Self-Organizing Fuzzy Neural Network
#

import keras as k
import numpy as np
import pandas as pd
import sklearn as sk

from keras import regularizers
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.metrics import mean_squared_error, \
                confusion_matrix, classification_report

# custom Fuzzy Layers
from notebooks.modeling.keras.layers import FuzzyLayer, DefuzzyLayer


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
        - input shape  : (*, features)
        - output shape : (neurons, 1)
    3 - Normalized Layer
            normalize each output of previous layer as
            relative amount from sum of all previous outputs
        - input shape  : (neurons, 1)
        - output shape : (neurons, 1)
    4 - Weighted Layer
            multiply bias vector (1+n_features,) by
            parameter vector (1+n_features,) of parameters
            from each fuzzy rule
            multiply each product by output of each rule's
            layer from normalized layer
        - input shape  : (1+features, 1) <- A
        - input shape  : (1+features, 1) <- B
        - output shape : (neurons, 1)
    5 - Output Layer
            summation of incoming signals from weighted layer
        - input shape  : (neurons, 1)
        - output shape : (1,)

    """

    def __init__(self, **kwargs):
        # initialize as Model object
        super().__init__(**kwargs)

        # initialize variables needed for processing
        pass
