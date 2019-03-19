#
# Class for Self-Organizing Fuzzy Neural Network
#

import keras as k
import numpy as np
import pandas as pd
import sklearn as sk

from keras import regularizers
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.metrics import mean_squared_error, \
                confusion_matrix, classification_report

# custom Fuzzy Layers
from notebooks.modeling.keras.layers import FuzzyLayer, DefuzzyLayer

class SOFNN(Sequential):
    """
    Class for building Self-Organizing Fuzzy Neural Networks

    -to house functions that optimize neural network
    """

    def __init__(self, **kwargs):
        # initialize as Sequential object
        super().__init__()

        # initialize variables needed for processing
        pass
