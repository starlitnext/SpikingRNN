# -*- coding=utf-8 -*-
# __author__ = 'xxq'

import theano
import theano.tensor as T
import numpy as np
from sklearn.base import BaseEstimator
import logging
import time
import os
import datetime
import cPickle as pickle
from data_utils import *

logger = logging.getLogger(__name__)


class Softmax(object):
    def __init__(self, input, n_in, n_out):
        W_init = np.asarray(np.random.uniform(size=(n_in, n_out),
                                         low=-.01, high=.01),
                       dtype=theano.config.floatX)
        self.W = theano.shared(value=W_init, name='W', borrow=True)
        b_init = np.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_init, name='b', borrow=True)

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

        self.input = input
        self.output = T.dot(input, self.W) + self.b

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    def __init__(self, input, n_in, n_out, rng=np.random.RandomState(1234), W=None, b=None, activation='sigmoid'):
        """

        :param rng: numpy.random.RandomState, a random number generator used to initialize weights
        :param input: theano.tensor.dmatrix, a symbolic tensor of shape (n_examples, n_in)
        :param n_in: int, dimensionality of input
        :param n_out: int, number of hidden units
        :param W:
        :param b:
        :param activation: Non linearity to be applied in the hidden layer
        """
        self.input = input
        if W is None:
            W_init = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

        if activation == 'sigmoid':
            W_init *= 4
            self.act_fn = T.nnet.sigmoid
        elif activation == 'tanh':
            self.act_fn = T.tanh
        elif activation == 'relu':
            self.act_fn = lambda x: x * (x > 0)
        else:
            raise NotImplementedError()

        W = theano.shared(value=W_init, name='W', borrow=True)

        if b is None:
            b_init = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_init, name='b', borrow=True)

        self.W = W
        self.b = b
        self.params = [self.W, self.b]

        lin_output = T.dot(input, self.W) + self.b
        self.output = self.act_fn(lin_output)


class RecurrentLayer(object):
    def __init__(self, input, n_in, n_out, activation='relu'):
        self.input = input
        if activation == 'sigmoid':
            self.act_fn = T.nnet.sigmoid
        elif activation == 'tanh':
            self.act_fn = T.tanh
        elif activation == 'relu':
            self.act_fn = lambda x: x * (x > 0)
        else:
            raise NotImplementedError()

        # recurrent weights as a shared variable
        W_re_init = np.asarray(np.random.uniform(size=(n_out, n_out),
                                                 low=-.01, high=.01),
                               dtype=theano.config.floatX)
        self.W_re = theano.shared(value=W_re_init, name='W_re', borrow=True)
        # input weights
        W_in_init = np.asarray(np.random.uniform(size=(n_in, n_out),
                                         low=-.01, high=.01),
                               dtype=theano.config.floatX)
        self.W_in = theano.shared(value=W_in_init, name='W_in', borrow=True)
        # neuron's initialize value
        h0_init = np.zeros((n_out,), dtype=theano.config.floatX)
        self.h0 = theano.shared(value=h0_init, name='h0', borrow=True)
        # bias
        bh_init = np.zeros((n_out,), dtype=theano.config.floatX)
        self.bh = theano.shared(value=bh_init, name='bh', borrow=True)

        self.params = [self.W_re, self.W_in, self.h0, self.bh]

        def step(x_t, h_tm1):
            h_t = self.act_fn(T.dot(x_t, self.W_in) + T.dot(h_tm1, self.W_re) + self.bh)
            return h_t

        self.output, _ = theano.scan(step,
                                     sequences=self.input,
                                     outputs_info=[self.h0,])

class MixedRNN(object):
    def __init__(self, input, n_in, n_hidden, n_recurrent, n_out):
        self.hiddenLayer = HiddenLayer(
            input=input,
            n_in=n_in,
            n_out=n_hidden
        )
        self.recurrentLayer = RecurrentLayer(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_recurrent,
        )
        self.softmaxLayer = Softmax(
            input=self.recurrentLayer.output,
            n_in=n_recurrent,
            n_out=n_out
        )

        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.recurrentLayer.W_re).sum() + abs(self.recurrentLayer.W_in).sum()
            + abs(self.softmaxLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.recurrentLayer.W_re ** 2).sum() + (self.recurrentLayer.W_in ** 2).sum()
            + (self.softmaxLayer.W ** 2).sum()
        )

        self.negative_log_liklihood = self.softmaxLayer.negative_log_likelihood
        self.errors = self.softmaxLayer.errors
        self.loss = lambda y: self.negative_log_liklihood(y)

        # self.params = self.hiddenLayer.params + self.recurrentLayer.params + self.softmaxLayer.params
        self.params = [self.hiddenLayer.W, self.recurrentLayer.W_re, self.recurrentLayer.W_in, self.softmaxLayer.W]

        # for every parameter, we maintain it's last update
        # the idea here is to use "momentum"
        # keep moving mostly in the same direction
        self.updates = {}
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            self.updates[param] = theano.shared(init)

        self.input = input


class MetaRNN(BaseEstimator):
    def __init__(self, n_in=5, n_hidden=50, n_recurrent=50, n_out=5, learning_rate=0.01,
                 n_epochs=100, L1_reg=0.00, L2_reg=0.00, learning_rate_decay=1.,
                 final_momentum=0.9, initial_momentum=0.5,
                 momentum_switchover=5):
        self.n_in = int(n_in)
        self.n_hidden = int(n_hidden)
        self.n_recurrent = int(n_recurrent)
        self.n_out = int(n_out)
        self.learning_rate = float(learning_rate)
        self.learning_rate_decay = float(learning_rate_decay)
        self.n_epochs = int(n_epochs)
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)
        self.initial_momentum = float(initial_momentum)
        self.final_momentum = float(final_momentum)
        self.momentum_switchover = int(momentum_switchover)

        self.ready()

    def ready(self):
        # input (where first dimension is time)
        self.x = T.matrix()
        # target (where first dimension is time)
        self.y = T.vector(name='y', dtype='int32')
        # initial hidden state of the RNN
        self.h0 = T.vector()
        # learning rate
        self.lr = T.scalar()

        self.rnn = MixedRNN(input=self.x, n_in=self.n_in,
                            n_hidden=self.n_hidden, n_recurrent=self.n_recurrent,
                            n_out=self.n_out)

        self.predict_proba = theano.function(inputs=[self.x, ],
                                             outputs=self.rnn.softmaxLayer.p_y_given_x)
        self.predict = theano.function(inputs=[self.x, ],
                                       outputs=self.rnn.softmaxLayer.y_pred)
        self.activation = theano.function(inputs=[self.x, ],
                                          outputs=self.rnn.softmaxLayer.output)
        self.spike_prob = theano.function(inputs=[self.x, ],
                                          outputs=self.rnn.hiddenLayer.output)

    def shared_dataset(self, data_xy):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX))

        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX))

        return shared_x, T.cast(shared_y, 'int32')

    def __getstate__(self):
        params = self.get_params()  # parameters set in constructor
        weights = [p.get_value() for p in self.rnn.params]
        state = (params, weights)
        return state

    def _set_weights(self, weights):
        i = iter(weights)

        for param in self.rnn.params:
            param.set_value(i.next())

    def __setstate__(self, state):
        params, weights = state
        self.set_params(**params)
        self.ready()
        self._set_weights(weights)

    def save(self, fpath='.', fname=None):
        """ Save a pickled representation of Model state. """
        fpathstart, fpathext = os.path.splitext(fpath)
        if fpathext == '.pkl':
            # User supplied an absolute path to a pickle file
            fpath, fname = os.path.split(fpath)

        elif fname is None:
            # Generate filename based on date
            date_obj = datetime.datetime.now()
            date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
            class_name = self.__class__.__name__
            fname = '%s.%s.pkl' % (class_name, date_str)

        fabspath = os.path.join(fpath, fname)

        logger.info("Saving to %s ..." % fabspath)
        file = open(fabspath, 'wb')
        state = self.__getstate__()
        pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()

    def load(self, path):
        """ Load model parameters from path. """
        logger.info("Loading from %s ..." % path)
        file = open(path, 'rb')
        state = pickle.load(file)
        self.__setstate__(state)
        file.close()

    def fit(self, X_train, Y_train, X_test=None, Y_test=None,
            validation_frequency=100):
        """ Fit model

        Pass in X_test, Y_test to compute test error and report during
        training.

        X_train : ndarray (n_seq x n_steps x n_in)
        Y_train : ndarray (n_seq x n_steps x n_out)

        validation_frequency : int
            in terms of number of sequences (or number of weight updates)
        """
        if X_test is not None:
            assert(Y_test is not None)
            self.interactive = True
            test_set_x, test_set_y = self.shared_dataset((X_test, Y_test))
        else:
            self.interactive = False

        train_set_x, train_set_y = self.shared_dataset((X_train, Y_train))
        self.train_set_x = train_set_x
        self.train_set_y = train_set_y

        n_train = train_set_x.get_value(borrow=True).shape[0]
        if self.interactive:
            n_test = test_set_x.get_value(borrow=True).shape[0]

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        logger.info('... building the model')

        index = T.lscalar('index')    # index to a case
        # learning rate (may change)
        l_r = T.scalar('l_r', dtype=theano.config.floatX)
        mom = T.scalar('mom', dtype=theano.config.floatX)  # momentum

        cost = self.rnn.loss(self.y) \
            + self.L1_reg * self.rnn.L1 \
            + self.L2_reg * self.rnn.L2_sqr

        compute_train_error = theano.function(inputs=[index, ],
                                              outputs=self.rnn.loss(self.y),
                                              givens={
                                                  self.x: train_set_x[index],
                                                  self.y: train_set_y[index]})

        if self.interactive:
            compute_test_error = theano.function(inputs=[index, ],
                        outputs=self.rnn.loss(self.y),
                        givens={
                            self.x: test_set_x[index],
                            self.y: test_set_y[index]})

        # compute the gradient of cost with respect to theta = (W, W_in, W_out)
        # gradients on the weights using BPTT
        gparams = []
        for param in self.rnn.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        updates = {}
        for param, gparam in zip(self.rnn.params, gparams):
            weight_update = self.rnn.updates[param]
            upd = mom * weight_update - l_r * gparam
            updates[weight_update] = upd
            updates[param] = param + upd

        # compiling a Theano function `train_model` that returns the
        # cost, but in the same time updates the parameter of the
        # model based on the rules defined in `updates`
        train_model = theano.function(inputs=[index, l_r, mom],
                                      outputs=cost,
                                      updates=updates,
                                      givens={
                                          self.x: train_set_x[index],
                                          self.y: train_set_y[index]})

        ###############
        # TRAIN MODEL #
        ###############
        logger.info('... training')
        epoch = 0

        while (epoch < self.n_epochs):
            epoch += 1
            for idx in xrange(n_train):
                effective_momentum = self.final_momentum \
                               if epoch > self.momentum_switchover \
                               else self.initial_momentum
                example_cost = train_model(idx, self.learning_rate,
                                           effective_momentum)

                # iteration number (how many weight updates have we made?)
                # epoch is 1-based, index is 0 based
                iter = (epoch - 1) * n_train + idx + 1

                if iter % validation_frequency == 0:
                    # compute loss on training set
                    train_losses = [compute_train_error(i)
                                    for i in xrange(n_train)]
                    this_train_loss = np.mean(train_losses)

                    if self.interactive:
                        test_losses = [compute_test_error(i)
                                        for i in xrange(n_test)]
                        this_test_loss = np.mean(test_losses)

                        logger.info('epoch %i, seq %i/%i, tr loss %f '
                                    'te loss %f lr: %f' % \
                        (epoch, idx + 1, n_train,
                         this_train_loss, this_test_loss, self.learning_rate))
                    else:
                        logger.info('epoch %i, seq %i/%i, train loss %f '
                                    'lr: %f' % \
                                    (epoch, idx + 1, n_train, this_train_loss,
                                     self.learning_rate))

            self.learning_rate *= self.learning_rate_decay


def dataFromSun():

    def train(train_x, train_y, valid_x, valid_y):

        n_epochs = 200
        n_hidden = 50
        n_recurrent = 100
        n_in = 6
        n_steps = 140
        n_classes = 2
        n_out = n_classes  # restricted to single softmax per time step

        model = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_recurrent=n_recurrent, n_out=n_out,
                        learning_rate=0.0001, learning_rate_decay=0.99,
                        n_epochs=n_epochs)

        model.fit(train_x, train_y, valid_x, valid_y, validation_frequency=1000)

        model.save(fname='model.pkl')
        return model

    logging.basicConfig(level=logging.INFO)
    t0 = time.time()
    train_x, train_y, valid_x, valid_y, test_x, test_y = spilit_eeg_data(train_ratio=1.0, valid_ratio=0.0, test_ratio=0.0, fname='data/data5.mat')
    print train_x.shape, train_y.shape
    print valid_x.shape, valid_y.shape
    print test_x.shape, test_y.shape

    # model = train(train_x, train_y, valid_x, valid_y)

    model = MetaRNN()
    model.load('model_back1.pkl')
    # weights = [p.get_value(borrow=True) for p in model.rnn.params]
    # print weights[0]

    # training set
    seqs = xrange(train_x.shape[0])
    err_cnt = 0
    for seq_num in seqs:
        true_label = train_y[seq_num, -1]
        guess_label = model.predict(train_x[seq_num, :, :])[-1]
        # print 'true_label: ', true_label, '\tguess_label: ', guess_label
        if true_label != guess_label:
            err_cnt += 1
    print 'Total test: ', train_x.shape[0]
    print 'Test Accuracy: %f%%.' % (100*(train_x.shape[0]-err_cnt)/train_x.shape[0],)

    return None

    # valid set
    seqs = xrange(valid_x.shape[0])
    err_cnt = 0
    for seq_num in seqs:
        true_label = valid_y[seq_num, -1]
        guess_label = model.predict(valid_x[seq_num, :, :])[-1]
        # print 'true_label: ', true_label, '\tguess_label: ', guess_label
        if true_label != guess_label:
            err_cnt += 1
    print 'Total test: ', valid_x.shape[0]
    print 'Test Accuracy: %f%%.' % (100*(valid_x.shape[0]-err_cnt)/valid_x.shape[0],)

    # test set
    seqs = xrange(test_x.shape[0])
    err_cnt = 0
    for seq_num in seqs:
        true_label = test_y[seq_num, -1]
        guess_label = model.predict(test_x[seq_num, :, :])[-1]
        # print 'true_label: ', true_label, '\tguess_label: ', guess_label
        if true_label != guess_label:
            err_cnt += 1
    print 'Total test: ', test_x.shape[0]
    print 'Test Accuracy: %f%%.' % (100*(test_x.shape[0]-err_cnt)/test_x.shape[0],)


    import matplotlib.pyplot as plt
    seqs = xrange(5)
    plt.close('all')
    for seq_num in seqs:
        fig = plt.figure()
        ax1 = plt.subplot(311)
        plt.plot(test_x[seq_num])
        ax1.set_title('input')
        ax2 = plt.subplot(312)

        # blue line will represent true classes
        # true_targets = plt.step(xrange(10), test_y[seq_num, :10], marker='o')

        # show probabilities (in b/w) output by model
        # guess = model.predict_proba(test_x[seq_num, :10, :])
        # guessed_probs = plt.imshow(guess.T, interpolation='nearest',
        #                            cmap='gray')
        activations = model.activation(test_x[seq_num])
        plt.plot(activations)
        ax2.set_title('output activations')
        # plt.ylim([0, 1])
        ax3 = plt.subplot(313)
        probs = model.spike_prob(test_x[seq_num])
        plt.plot(probs[:, 0])
        plt.show()

    print "Elapsed time: %f" % (time.time() - t0)

def wrist_data():

    def train(train_x, train_y):

        n_epochs = 450
        n_hidden = 50
        n_recurrent = 60
        n_in = 14
        n_steps = 128
        n_classes = 3
        n_out = n_classes  # restricted to single softmax per time step

        model = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_recurrent=n_recurrent, n_out=n_out,
                        learning_rate=0.001, learning_rate_decay=0.999,
                        n_epochs=n_epochs)

        model.fit(train_x, train_y, validation_frequency=1000)

        model.save(fname='wrist_model.pkl')
        return model

    logging.basicConfig(level=logging.INFO)
    t0 = time.time()
    wrist_eeg = WristMovementEEG()
    train_x, train_y, test_x, test_y = wrist_eeg.spilit_data()
    print train_x.shape
    print train_y.shape
    print test_x.shape
    print test_y[:, 0]

    # model = train(train_x, train_y)

    model = MetaRNN()
    model.load('wrist_model.pkl')

    # test set
    test_x = train_x
    test_y = train_y
    seqs = xrange(30)
    err_cnt = 0
    for seq_num in seqs:
        true_label = int(test_y[seq_num, -1])
        guess_label = model.predict(test_x[seq_num, :, :])[-1]
        print 'true_label: ', true_label, '\tguess_label: ', guess_label
        if true_label != guess_label:
            err_cnt += 1
    print 'Total test: ', test_x.shape[0]
    print 'Test Accuracy: %f%%.' % (100*(test_x.shape[0]-err_cnt)/test_x.shape[0],)

    import matplotlib.pyplot as plt
    seqs = xrange(30)
    plt.close('all')
    for seq_num in seqs:
        fig = plt.figure()
        ax1 = plt.subplot(311)
        plt.plot(test_x[seq_num])
        ax1.set_title('input')
        ax2 = plt.subplot(312)

        # blue line will represent true classes
        # true_targets = plt.step(xrange(10), test_y[seq_num, :10], marker='o')

        # show probabilities (in b/w) output by model
        # guess = model.predict_proba(test_x[seq_num, :10, :])
        # guessed_probs = plt.imshow(guess.T, interpolation='nearest',
        #                            cmap='gray')
        activations = model.activation(test_x[seq_num])
        plt.plot(activations)
        plt.legend('0')
        ax2.set_title('output activations')
        # plt.ylim([0, 1])
        ax3 = plt.subplot(313)
        probs = model.spike_prob(test_x[seq_num])
        plt.plot(probs[:, 0])
        plt.show()

    print "Elapsed time: %f" % (time.time() - t0)


if __name__ == '__main__':
    dataFromSun()
    # wrist_data()