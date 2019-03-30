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

from sklearn.metrics import confusion_matrix, classification_report

# custom Fuzzy Layers
from .layers import \
    FuzzyLayer, NormalizedLayer, WeightedLayer


class SOFNN(object):
    """
    Self-Organizing Fuzzy Neural Network
    =======================================================

    -Implemented per description in:
        "An on-line algorithm for creating self-organizing
        fuzzy neural networks" - Leng, Prasad, McGinnity (2004)
    -Composed of 5 layers with varying "fuzzy rule" nodes

    * = samples

    Parameters
    ==========
    - X_train : training input data
        - shape :(train_*, features)
    - X_test  : testing input data
        - shape: (test_*, features)
    - y_train : training output data
        - shape: (train_*,)
    - y_test  : testing output data
        - shape: (test_*,)

    Layers
    ======
    1 - Input Layer
            input dataset
        - input shape  : (*, features)
    2 - Radial Basis Function Layer (Fuzzy Layer)
            layer to hold fuzzy rules for complex system
        - input : x
            shape: (*, features * neurons)
        - output : phi
            shape : (*, neurons)
    3 - Normalized Layer
            normalize each output of previous layer as
            relative amount from sum of all previous outputs
        - input : phi
            shape  : (*, neurons)
        - output : psi
            shape : (*, neurons)
    4 - Weighted Layer
            multiply bias vector (1+n_features, neurons) by
            parameter vector (1+n_features,) of parameters
            from each fuzzy rule
            multiply each product by output of each rule's
            layer from normalized layer
        - inputs : [x, psi]
            shape  : [(*, 1+features), (*, neurons)]
        - output : f
            shape : (*, neurons)
    5 - Output Layer
            summation of incoming signals from weighted layer
        - input shape  : (*, neurons)
        - output shape : (*,)

    Functions
    =========
    - system_error_check :
        - system error considers generalized performance of overall network
        - add neuron if error above predefined error threshold (delta)
    - if_part_check :
        - if-part criterion checks if current fuzzy rules cover/cluster input vector suitably
    - add_neuron :
        - add one neuron to model
    - prune_neuron :
        - remove neuron from model
    - combine_membership_functions :
        - combine similar membership functions
    """

    def __init__(self, X_train, X_test, y_train, y_test, neurons=1):

        # set data attributes
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test
        # set initial number of neurons
        self._neurons = neurons

        # build model on init
        self._model = self._build_model()

    def _build_model(self):
        """
        Create and compile model
        """

        print('Building SOFNN with {} neurons'.format(self._neurons))

        # get shape of training data
        samples, feats = self._X_train.shape

        # add layers
        inputs = Input(name='Inputs', shape=(feats,))
        fuzz = FuzzyLayer(self._neurons)
        norm = NormalizedLayer(self._neurons)
        weights = WeightedLayer(self._neurons)

        # run through layers
        phi = fuzz(inputs)
        psi = norm(phi)
        f = weights([inputs, psi])
        raw_output = Dense(1, name='RawOutput', activation='linear', use_bias=False)(f)
        preds = Activation(name='OutputActivation', activation='relu')(raw_output)

        # compile model and output summary
        model = Model(inputs=inputs, outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mape'])
        print(model.summary())

        return model

    def train_model(self, epochs=50, batch_size=None):
        """
        Run currently saved model
        """
        # fit model and evaluate
        self._model.fit(self._X_train, self._y_train,
                        epochs=epochs, batch_size=batch_size, shuffle=False)

    def evaluate_model(self, threshold=0.5):
        """
        Evaluate currently trained model
        """
        scores = self._model.evaluate(self._X_test, self._y_test, verbose=1)
        accuracy = scores[1] * 100
        print("\nAccuracy: {:.2f}%".format(accuracy))

        # print confusion matrix
        print('\nConfusion Matrix')
        print('=' * 20)
        y_pred = np.squeeze(np.where(
            self._model.predict(self._X_test) >= threshold, 1, 0), axis=-1)
        print(pd.DataFrame(confusion_matrix(self._y_test, y_pred),
                           index=['true:no', 'true:yes'], columns=['pred:no', 'pred:yes']))

        # print classification report
        print('\nClassification Report')
        print('=' * 20)
        print(classification_report(self._y_test, y_pred, labels=[0, 1]))

        # return predicted values
        return y_pred

    @staticmethod
    def loss_function(y_true, y_pred):
        """
        Custom loss function

        E = exp{-sum[i=1,j; 1/2 * [pred(j) - test(j)]^2]}
        """
        return K.sum(1/2 * K.square(y_pred - y_true))
