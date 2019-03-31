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
    ====================================

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
    - neurons : number of initial neurons
    - debug : debug flag

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

    Methods
    =======
    - self_organize :
        - run main logic to organize FNN
    - add_neuron :
        - add one neuron to model
    - prune_neuron :
        - remove neuron from model
    - combine_membership_functions :
        - combine similar membership functions

    Secondary Methods
    =================
    - build_model :
        - build and compile model
    - train_model :
        - train on data
    - evaluate_model :
        - evaluate model on test data
    - get_layer :
        - return layer object from model by name
    - get_layer_weights :
        - get current weights from any layer in model
    - get_layer_output :
        - get test output from any layer in model
    - loss_function :
        - custom loss function per Leng, Prasad, McGinnity (2004)
    - error_criterion :
        - considers generalized performance of overall network
        - add neuron if error above predefined error threshold (delta)
    - if_part_criterion :
        - checks if current fuzzy rules cover/cluster input vector suitably

    """

    def __init__(self, X_train, X_test, y_train, y_test, neurons=1, debug=True):
        # set debug flag
        self.debug = debug

        # set data attributes
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test
        # set initial number of neurons
        self._neurons = neurons

        # build model on init
        self._model = self._build_model()

    def self_organize(self, epochs=50, batch_size=None, eval_thresh=0.5,
                      ifpart_thresh=0.1354, delta=0.12,
                      ksig=1.12, max_widens=250):
        """
        Main run function to handle organization logic

        - Train initial model in parameters then begin self-organization
        - If fails If-Part test, widen rule widths
        - If still fails, reset to original widths
            then add neuron and retrain weights

        Parameters
        ==========
        epochs : int
            - number of training epochs
        batch_size : int
            - size of training batch
        eval_thresh : float
            - cutoff for 0/1 class
        ifpart_thresh : float
            - threshold for if-part
        ksig : float
            - factor to widen centers
        max_iters : int
            - max iterations for widening centers
        """
        # initial training of model - yields predictions
        self._train_model(epochs=epochs, batch_size=batch_size)
        y_pred = self._evaluate_model(eval_thresh=eval_thresh)

        # run criterion checks and organize accordingly
        self.run_criterion_checks(y_pred=y_pred, ifpart_thresh=ifpart_thresh,
                                  ksig=ksig, max_widens=max_widens, delta=delta)

    def run_criterion_checks(self, y_pred, ifpart_thresh=0.1354,
                             ksig=1.12, max_widens=250, delta=0.12):
        """
        Run logic to check on system error or if-part criteron

        Parameters
        ==========
        y_pred : array
            - predictions
        ifpart_thresh : float
            - threshold for if-part
        ksig : float
            - factor to widen centers
        max_iters : int
            - max iterations for widening centers
        delta : float
            - threshold for error criterion whether new neuron to be added
        """

        # get old weights and create current weight vars
        start_weights = self._get_layer('FuzzyRules').get_weights()

        # widen centers if necessary
        if not self._if_part_criterion(ifpart_thresh=ifpart_thresh):
            self.widen_centers(ksig=ksig, max_widens=max_widens)
        if not self._error_criterion(y_pred=y_pred, delta=delta):
            self._get_layer('FuzzyRules').set_weights(start_weights)
            self.add_neuron()

    def add_neuron(self):
        """
        Rebuild model but with extra neuron
        """
        if self.debug:
            print('Adding neuron...')
        pass

    def widen_centers(self, ksig=1.12, max_widens=250):
        """
        Widen center of neurons to better cover data

        Parameters
        ==========
        ksig : float
            - multiplicative factor widening centers
        max_iters : int
            - number of max iterations to update centers before ending
        """
        # print alert of successful widening
        if self.debug:
            print('Widening centers...')

        # get fuzzy layer and output to find max neuron output
        fuzz = self._get_layer('FuzzyRules')

        # get old weights and create current weight vars
        weights = fuzz.get_weights()
        c, s = weights[0], weights[1]

        # repeat until if-part criterion satisfied
        # only perform for max iterations
        counter = 0
        while not self._if_part_criterion():

            counter += 1
            # check if max iterations exceeded
            if counter > max_widens:
                if self.debug:
                    print('Max iterations reached - resetting weights')
                return False

            # get neuron with max-output for each sample
            # then select the most common one to update
            fuzz_out = self._get_layer_output('FuzzyRules')
            maxes = np.argmax(fuzz_out, axis=-1)
            max_neuron = np.argmax(np.bincount(maxes.flat))

            # select minimum width to expand
            # and multiply by factor
            mf_min = s[:, max_neuron].argmin()
            s[mf_min, max_neuron] = ksig * s[mf_min, max_neuron]
            # update weights

            new_weights = [c, s]
            fuzz.set_weights(new_weights)

        # print alert of successful widening
        if self.debug:
            print('Centers widened after {} iterations'.format(counter))

    def _build_model(self):
        """
        Create and compile model
        """

        if self.debug:
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
        preds = Activation(name='OutputActivation', activation='sigmoid')(raw_output)

        # compile model and output summary
        model = Model(inputs=inputs, outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mape'])
        if self.debug:
            print(model.summary())

        return model

    def _train_model(self, epochs=50, batch_size=None):
        """
        Run currently saved model

        Parameters
        ==========
        epochs : int
            - number of training epochs
        batch_size : int
            - size of training batch
        """
        # fit model and evaluate
        self._model.fit(self._X_train, self._y_train,
                        epochs=epochs, batch_size=batch_size, shuffle=False)

    def _evaluate_model(self, eval_thresh=0.5):
        """
        Evaluate currently trained model

        Parameters
        ==========
        eval_thresh : float
            - cutoff threshold for positive/negative classes
        """
        scores = self._model.evaluate(self._X_test, self._y_test, verbose=1)
        accuracy = scores[1] * 100
        print("\nAccuracy: {:.2f}%".format(accuracy))

        # print confusion matrix
        print('\nConfusion Matrix')
        print('=' * 20)
        y_pred = np.squeeze(np.where(
            self._model.predict(self._X_test) >= eval_thresh, 1, 0), axis=-1)
        print(pd.DataFrame(confusion_matrix(self._y_test, y_pred),
                           index=['true:no', 'true:yes'], columns=['pred:no', 'pred:yes']))

        # print classification report
        print('\nClassification Report')
        print('=' * 20)
        print(classification_report(self._y_test, y_pred, labels=[0, 1]))

        # return predicted values
        return y_pred

    def _error_criterion(self, y_pred, delta=0.12):
        """
        Check error criterion for neuron-adding process
            - return True if no need to grow neuron
            - return False if above threshold and need to add neuron

        Parameters
        ==========
        y_pred : array
            - predictions
        delta : float
            - threshold for error criterion whether new neuron to be added
        """
        # mean of absolute test difference
        return np.abs(y_pred - self._y_test).mean() <= delta

    def _if_part_criterion(self, ifpart_thresh=0.1354):
        """
        Check if-part criterion for neuron adding process
            - for each sample, get max of all neuron outputs (pre-normalization)
            - test whether max val at or above threshold

        Parameters
        ==========
        ifpart_thresh : float
            - threshold for if-part detections
        """
        # get max val
        fuzz_out = self._get_layer_output('FuzzyRules')
        # check if max neuron output is above threshold
        maxes = np.max(fuzz_out, axis=-1) >= ifpart_thresh
        # return True if at least half of samples agree
        return (maxes.sum() / len(maxes)) >= 0.5

    def _get_layer(self, layer=None):
        """
        Get layer object based on input parameter
            - exception of Input layer

        Parameters
        ==========
        layer : str or int
            - layer to get weights from
            - input can be layer name or index
        """
        # if named parameter
        if layer in [mlayer.name for mlayer in self._model.layers[1:]]:
            layer_out = self._model.get_layer(layer)
        # if indexed parameter
        elif layer in range(1, len(self._model.layers)):
            layer_out = self._model.layers[layer]
        else:
            raise ValueError('Error: layer must be layer name or index')
        return layer_out

    def _get_layer_weights(self, layer=None):
        """
        Get weights of layer based on input parameter
            - exception of Input layer

        Parameters
        ==========
        layer : str or int
            - layer to get weights from
            - input can be layer name or index
        """
        return self._get_layer(layer).get_weights()

    def _get_layer_output(self, layer=None):
        """
        Get output of layer based on input parameter
            - exception of Input layer

        Parameters
        ==========
        layer : str or int
            - layer to get test output from
            - input can be layer name or index
        """
        last_layer = self._get_layer(layer)
        intermediate_model = Model(inputs=self._model.input,
                                   outputs=last_layer.output)
        return intermediate_model.predict(self._X_test)

    @staticmethod
    def _loss_function(y_true, y_pred):
        """
        Custom loss function

        E = exp{-sum[i=1,j; 1/2 * [pred(j) - test(j)]^2]}

        Parameters
        ==========
        y_true : array
            - true values
        y_pred : array
            - predicted values
        """
        return K.sum(1 / 2 * K.square(y_pred - y_true))
