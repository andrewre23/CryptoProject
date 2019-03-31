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
import matplotlib.pyplot as plt

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Activation

from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

# custom Fuzzy Layers
from .layers import FuzzyLayer, NormalizedLayer, WeightedLayer, OutputLayer


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

    Methods
    =======
    - build_model :
        - build and compile model
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

    def __init__(self, X_train, X_test, y_train, y_test, neurons=1, s_init=4, debug=True):
        # set debug flag
        self.__debug = debug

        # set data attributes
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test
        # set initial number of neurons
        self.__neurons = neurons

        # build model on init
        self.build_model()
        if self.__neurons == 1:
            self.__initialize_model(s_init=s_init)

    def build_model(self):
        """
        Create and compile model
        - sets compiled model as self.model

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
        """

        if self.__debug:
            print('\nBUILDING SOFNN WITH {} NEURONS'.format(self.__neurons))

        # get shape of training data
        samples, feats = self._X_train.shape

        # add layers
        inputs = Input(name='Inputs', shape=(feats,))
        fuzz = FuzzyLayer(self.__neurons)
        norm = NormalizedLayer(self.__neurons)
        weights = WeightedLayer(self.__neurons)
        raw = OutputLayer()

        # run through layers
        phi = fuzz(inputs)
        psi = norm(phi)
        f = weights([inputs, psi])
        raw_output = raw(f)
        # raw_output = Dense(1, name='RawOutput', activation='linear', use_bias=False)(f)
        preds = Activation(name='OutputActivation', activation='sigmoid')(raw_output)

        # compile model and output summary
        model = Model(inputs=inputs, outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'mape'])
        if self.__debug:
            print(model.summary())

        self.model = model

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
        delta : float
            - threshold for error criterion whether new neuron to be added
        ksig : float
            - factor to widen centers
        max_widens : int
            - max iterations for widening centers
        """
        # initial training of model - yields predictions
        if self.__debug:
            print('Beginning model training...')
        self._train_model(epochs=epochs, batch_size=batch_size)
        if self.__debug:
            print('Initial Model Evaluation')
        y_pred = self._evaluate_model(eval_thresh=eval_thresh)

        # run update logic until passes criterion checks
        while not self.error_criterion(y_pred, delta=delta) and \
                not self.if_part_criterion(ifpart_thresh=ifpart_thresh):
            # run criterion checks and organize accordingly
            self.organize(y_pred=y_pred, ifpart_thresh=ifpart_thresh,
                          ksig=ksig, max_widens=max_widens, delta=delta)
            if self.__debug:
                self._evaluate_model(eval_thresh=eval_thresh)

        if self.__debug:
            print('\nSelf-Organization complete!')
            print('If-Part and Error Criterion satisfied')
            print('Final Evaluation')
            self._evaluate_model(eval_thresh=eval_thresh)

    def organize(self, y_pred, ifpart_thresh=0.1354,
                 ksig=1.12, max_widens=250, delta=0.12):
        """
        Run logic to check on system error or if-part criteron

        Parameters
        ==========
        y_pred : np.array
            - predictions
        ifpart_thresh : float
            - threshold for if-part
        ksig : float
            - factor to widen centers
        max_widens : int
            - max iterations for widening centers
        delta : float
            - threshold for error criterion whether new neuron to be added
        """

        # get copy of initial fuzzy weights
        start_weights = self._get_layer_weights('FuzzyRules')

        # widen centers if necessary
        if not self.if_part_criterion(ifpart_thresh=ifpart_thresh):
            self.widen_centers(ksig=ksig, max_widens=max_widens)
        # add neuron if necessary
        if not self.error_criterion(y_pred=y_pred, delta=delta):
            # reset fuzzy weights if previously widened
            if not np.array_equal(start_weights, self._get_layer_weights('FuzzyRules')):
                self._get_layer('FuzzyRules').set_weights(start_weights)
            self.add_neuron()

    def add_neuron(self):
        """
        Add extra neuron to model while
        keeping current neuron weights
        """
        if self.__debug:
            print('\nAdding neuron...')

        # get current weights
        c_curr, s_curr = self._get_layer_weights('FuzzyRules')

        # get weights for new neuron
        ck, sk = self._new_neuron_weights()
        # expand dim for stacking
        ck = np.expand_dims(ck, axis=-1)
        sk = np.expand_dims(sk, axis=-1)
        c_new = np.hstack((c_curr, ck))
        s_new = np.hstack((s_curr, sk))

        # increase neurons and rebuild model
        self.__neurons += 1
        self.build_model()

        # update weights
        new_weights = [c_new, s_new]
        self._get_layer('FuzzyRules').set_weights(new_weights)

        # validate weights updated as expected
        final_weights = self._get_layer_weights('FuzzyRules')
        assert np.allclose(new_weights[0], final_weights[0])
        assert np.allclose(new_weights[1], final_weights[1])

    def prune_neurons(self, tol_lim=0.8):
        """
        Prune any unimportant neurons per effect on RMSE
        """
        if self.__debug:
            print('\nPruning neurons...')
        # calculate RMSE
        # rmse =

    def widen_centers(self, ksig=1.12, max_widens=250):
        """
        Widen center of neurons to better cover data

        Parameters
        ==========
        ksig : float
            - multiplicative factor widening centers
        max_widens : int
            - number of max iterations to update centers before ending
        """
        # print alert of successful widening
        if self.__debug:
            print('\nWidening centers...')

        # get fuzzy layer and output to find max neuron output
        fuzz = self._get_layer('FuzzyRules')

        # get old weights and create current weight vars
        c, s = fuzz.get_weights()

        # repeat until if-part criterion satisfied
        # only perform for max iterations
        counter = 0
        while not self.if_part_criterion():

            counter += 1
            # check if max iterations exceeded
            if counter > max_widens:
                if self.__debug:
                    print('Max iterations reached ({})'
                          .format(counter - 1))
                return False
            if self.__debug and counter % 20 == 0:
                print('Iteration {}'.format(counter))

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
        if self.__debug:
            print('Centers widened after {} iterations'.format(counter))

    def error_criterion(self, y_pred, delta=0.12):
        """
        Check error criterion for neuron-adding process
            - return True if no need to grow neuron
            - return False if above threshold and need to add neuron

        Parameters
        ==========
        y_pred : np.array
            - predictions
        delta : float
            - threshold for error criterion whether new neuron to be added
        """
        # mean of absolute test difference
        return np.abs(y_pred - self._y_test).mean() <= delta

    def if_part_criterion(self, ifpart_thresh=0.1354):
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

    def __initialize_model(self, s_init=4):
        """
        Initialize neuron weights

        c_init = Average(X).T
        s_init = s_init

        """
        # derive initial c and s
        c_init = self._X_train.values.mean(axis=0, keepdims=True).T
        s_init = np.repeat(s_init, c_init.size).reshape(c_init.shape)
        start_weights = [c_init, s_init]
        self._get_layer('FuzzyRules').set_weights(start_weights)

        # validate weights updated as expected
        final_weights = self._get_layer_weights('FuzzyRules')
        assert np.allclose(start_weights[0], final_weights[0])
        assert np.allclose(start_weights[1], final_weights[1])

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
        self.model.fit(self._X_train, self._y_train, verbose=0,
                       epochs=epochs, batch_size=batch_size, shuffle=False)

    def _evaluate_model(self, eval_thresh=0.5):
        """
        Evaluate currently trained model

        Parameters
        ==========
        eval_thresh : float
            - cutoff threshold for positive/negative classes

        Returns
        =======
        y_pred : np.array
            - predicted values
            - shape: (samples,)
        """
        # calculate accuracy scores
        scores = self.model.evaluate(self._X_test, self._y_test, verbose=1)
        raw_preds = self.model.predict(self._X_test)
        y_pred = np.squeeze(np.where(raw_preds >= eval_thresh, 1, 0), axis=-1)

        # get prediction scores and prediction
        accuracy = scores[1]
        auc = roc_auc_score(self._y_test, raw_preds)

        # print accuracy and AUC score
        print('\nAccuracy Measures')
        print('=' * 20)
        print("Accuracy:  {:.2f}%".format(100 * accuracy))
        print("AUC Score: {:.2f}%".format(100 * auc))

        # print confusion matrix
        print('\nConfusion Matrix')
        print('=' * 20)
        print(pd.DataFrame(confusion_matrix(self._y_test, y_pred),
                           index=['true:no', 'true:yes'], columns=['pred:no', 'pred:yes']))

        # print classification report
        print('\nClassification Report')
        print('=' * 20)
        print(classification_report(self._y_test, y_pred, labels=[0, 1]))

        self._plot_results(y_pred=y_pred)
        # return predicted values
        return y_pred

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
        if layer in [mlayer.name for mlayer in self.model.layers[1:]]:
            layer_out = self.model.get_layer(layer)
        # if indexed parameter
        elif layer in range(1, len(self.model.layers)):
            layer_out = self.model.layers[layer]
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
        intermediate_model = Model(inputs=self.model.input,
                                   outputs=last_layer.output)
        return intermediate_model.predict(self._X_test)

    def _min_dist_vector(self):
        """
        Get minimum distance vector

        Returns
        =======
        min_dist : np.array
            - average minimum distance vector across samples
            - shape: (features, neurons)
        """
        # get input values and fuzzy weights
        x = self._X_train.values
        samples = x.shape[0]
        c = self._get_layer_weights('FuzzyRules')[0]

        # align x and c and assert matching dims
        aligned_x = x.repeat(self.__neurons). \
            reshape(x.shape + (self.__neurons,))
        aligned_c = c.repeat(samples).reshape((samples,) + c.shape)
        assert aligned_x.shape == aligned_c.shape

        # average the minimum distance across samples
        return np.abs(aligned_x - aligned_c).mean(axis=0)

    def _new_neuron_weights(self, dist_thresh=1):
        """
        Return new c and s weights for k new fuzzy neuron

        Parameters
        ==========
        dist_thresh : float
            - multiplier of average features values to use as distance thresholds

        Returns
        =======
        ck : np.array
            - average minimum distance vector across samples
            - shape: (features,)
        sk : np.array
            - average minimum distance vector across samples
            - shape: (features,)
        """
        # get input values and fuzzy weights
        x = self._X_train.values
        c, s = self._get_layer_weights('FuzzyRules')

        # get minimum distance vector
        min_dist = self._min_dist_vector()
        # get minimum distance across neurons
        # and arg-min for neuron with lowest distance
        dist_vec = min_dist.min(axis=-1)
        min_neurs = min_dist.argmin(axis=-1)

        # get min c and s weights
        c_min = c[:, min_neurs].diagonal()
        s_min = s[:, min_neurs].diagonal()
        assert c_min.shape == s_min.shape

        # set threshold distance as factor of mean
        # value for each feature across samples
        kd_i = x.mean(axis=0) * dist_thresh

        # get final weight vectors
        ck = np.where(dist_vec <= kd_i, c_min, x.mean(axis=0))
        sk = np.where(dist_vec <= kd_i, s_min, dist_vec)
        return ck, sk

    def _plot_results(self, y_pred):
        """
        Plot predictions against time series

        Parameters
        ==========
        y_pred : np.array
            - predicted values
        """
        # plotting results
        df_plot = pd.DataFrame()

        # create pred/true time series
        df_plot['price'] = self._X_test['bitcoin_close']
        df_plot['pred'] = y_pred * df_plot['price']
        df_plot['true'] = self._y_test * df_plot['price']
        df_plot['hits'] = df_plot['price'] * (df_plot['pred'] == df_plot['true'])
        df_plot['miss'] = df_plot['price'] * (df_plot['pred'] != df_plot['true'])

        fig, ax = plt.subplots(figsize=(12, 8))
        plt.plot(df_plot['price'], color='b')
        plt.bar(df_plot['price'].index, df_plot['hits'], color='g')
        plt.bar(df_plot['price'].index, df_plot['miss'], color='r')
        for label in ax.xaxis.get_ticklabels()[::400]:
            label.set_visible(False)

        plt.title('BTC Close Price Against Predictions')
        plt.xlabel('Dates')
        plt.ylabel('BTC Price ($)')
        plt.grid(True)
        plt.xticks(df_plot['price'].index[::4],
                   df_plot['price'].index[::4], rotation=70)
        plt.show()

    @staticmethod
    def _loss_function(y_true, y_pred):
        """
        Custom loss function

        E = exp{-sum[i=1,j; 1/2 * [pred(j) - test(j)]^2]}

        Parameters
        ==========
        y_true : np.array
            - true values
        y_pred : np.array
            - predicted values
        """
        return K.sum(1 / 2 * K.square(y_pred - y_true))
