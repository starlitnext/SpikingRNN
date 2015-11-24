# -*- coding=utf-8 -*-
# __author__ = 'xxq'

import pyNN.nest as sim
import numpy as np
import time
from data_utils import *
from nets import *
import matplotlib.pyplot as plt
import scipy.io as sio


class RecurrentSpikingNN(object):
    def __init__(self, sim_time, n_spike_source, n_hidden_neurons, n_output_neurons, W_in, W_re, W_out):
        self.sim_time = sim_time
        self.n_spike_source = int(n_spike_source)
        self.n_hidden_neurons = int(n_hidden_neurons)
        self.n_output_neurons = int(n_output_neurons)
        self.mem_scale = 2**16                  ## new new new
        self.weight_scale = 2**15               ## new new new
        self.parameters = {
            'VBits': 32,
            'VdecBits': 32,
            'Vdec_sig': 0,
            'V_th': 2000 * self.mem_scale / 1000, ## new new new
            'wBits': 16,
            'wAccBits': 20,
            'wAccShift': 1,
            't_ref': 0.
        }
        self.cell_type = sim.native_cell_type('iaf_psc_delta_int')
        self.parameters_out = {
            u'E_L': 0.0,
            u'I_e': 1.0,  # 用这个参数来表示leaky, 1.0则表示没有leak
            u'V_reset': 0.0,
            u'V_th': 2000, # 10000,                 ## new new new
            u't_ref': 0.0,
        }
        self.cell_type_out = sim.native_cell_type('iaf_psc_delta_xxq')
        # network structure
        self.spike_source = None
        self.hidden_neurons = None
        self.output_neurons = None
        self.connection_in = None
        self.connection_hidden = None
        self.connection_out = None
        W_in = np.round(W_in * self.weight_scale) / 1000.
        W_re = np.round(W_re * self.weight_scale) / 1000.
        # print W_in

        self.hidden_spike_cnts = []

        self.connection_in_list, self.connection_hidden_list, self.connection_out_list = self.getListConnection(W_in, W_re, W_out)
        self.ready()

    def getListConnection(self, W_in, W_re, W_out):
        assert (W_in.shape[0] == self.n_spike_source and W_in.shape[1] == self.n_hidden_neurons
                and W_re.shape[0] == self.n_hidden_neurons and W_re.shape[1] == self.n_hidden_neurons
                and W_out.shape[0] == self.n_hidden_neurons and W_out.shape[1] == self.n_output_neurons), \
                'neuron numbers and weight matrix don\'t match!!!'
        connection_in_list = []
        for i in range(self.n_spike_source):
            for j in range(self.n_hidden_neurons):
                connection_in_list.append((i, j, W_in[i, j], 0.01)) # no delay, may be raise a error
        connection_hidden_list = []
        for i in range(self.n_hidden_neurons):
            for j in range(self.n_hidden_neurons):
                connection_hidden_list.append((i, j, W_re[i, j], 1.)) # delay should be equal to time step
        connection_out_list = []
        for i in range(self.n_hidden_neurons):
            for j in range(self.n_output_neurons):
                connection_out_list.append((i, j, W_out[i, j], 0.01)) # no delay, may be raise a error
        return connection_in_list, connection_hidden_list, connection_out_list


    def ready(self):
        sim.setup(timestep=1.)

        # set up spike sources
        self.spike_source = sim.Population(self.n_spike_source, sim.SpikeSourceArray(), label='spike sources')
        # set up hidden neurons
        self.hidden_neurons = sim.Population(self.n_hidden_neurons, self.cell_type(**self.parameters), label='hidden neurons')
        # set up output neurons
        self.output_neurons = sim.Population(self.n_output_neurons, self.cell_type_out(**self.parameters_out), label='output neurons')
        self.output_neurons.set(I_e=1.0, V_th=1500.0) ## new new new

        # build connections
        self.connection_in = sim.Projection(self.spike_source, self.hidden_neurons, sim.FromListConnector(self.connection_in_list))
        self.connection_hidden = sim.Projection(self.hidden_neurons, self.hidden_neurons, sim.FromListConnector(self.connection_hidden_list))
        self.connection_out = sim.Projection(self.hidden_neurons, self.output_neurons, sim.FromListConnector(self.connection_out_list))

        self.output_neurons.record('spikes')
        self.hidden_neurons.record('spikes')

    def run(self, spiketimes):
        assert spiketimes.shape[0] == self.n_spike_source, 'spiketimes length should be equal to input neurons'
        start = time.clock()
        sim.reset()
        end = time.clock()
        print "reset uses %f s." % (end - start)
        for i in range(self.n_spike_source):
            spiketime = np.array(spiketimes[i], dtype=float)
            if spiketimes[i].any():
                self.spike_source[i].spike_times = spiketime

        sim.initialize(self.hidden_neurons, V_m=0)
        sim.initialize(self.output_neurons, V_m=0.)
        sim.run(self.sim_time)

        spiketrains = self.output_neurons.get_data(clear=True).segments[0].spiketrains

        # vtrace = self.hidden_neurons.get_data(clear=True).segments[0].filter(name='V_m')[0]
        # plt.figure()
        # plt.plot(vtrace.times, vtrace)
        # plt.show()

        hidden_spiketrains = self.hidden_neurons.get_data(clear=True).segments[0].spiketrains
        spike_cnts = 0
        for spiketrain in hidden_spiketrains:
            spike_cnts += len(list(np.array(spiketrain)))
        self.hidden_spike_cnts.append(spike_cnts)
        print 'hidden spikes: ', spike_cnts

        spiketimes_out = []
        for spiketrain in spiketrains:
            spiketimes_out.append(list(np.array(spiketrain)))


        return np.array(spiketimes_out)

    def end(self):
        sim.end()


class ConvertRNN2SNN(object):
    def __init__(self, sim_time, rnn_model):
        self.sim_time = sim_time
        self.rnn_model = rnn_model
        self.W, self.W_re, self.W_in, self.W_out = [p.get_value(borrow=True) for p in rnn_model.rnn.params]
        self.n_input = self.W.shape[1] # 50
        self.n_hidden = self.W_re.shape[0] # 100
        self.n_out = self.W_out.shape[1] # 2
        self.snn_model = RecurrentSpikingNN(sim_time, self.n_input, self.n_hidden, self.n_out, self.W_in, self.W_re, self.W_out)

    def getSpiketimes(self, X):
        """

        :param X: input , shape: 140, 60
        :return: spiketimes: shape (50,)
        """
        spike_probs = self.rnn_model.spike_prob(X) # shape: 140, 50 equal to : spike_probs = X.dot(W)
        probs = np.random.rand(spike_probs.shape[0], spike_probs.shape[1])
        spike_state = spike_probs > probs # 140, 50
        spiketimes = []
        for i in range(spike_probs.shape[1]):
            spiketimes.append(np.where(spike_state[:, i])[0] + 1) # add one because simulator doesn't support spiketime to be zero

        return np.array(spiketimes)

    def predictSingleCase(self, X):
        """

        :param X: (140, 6) input data
        :return: pridected label
        """
        spiketimes = self.getSpiketimes(X) # shape: 50,
        output_spiketimes = self.snn_model.run(spiketimes)
        spike_cnt = []
        for i in range(self.n_out):
            spike_cnt.append(len(output_spiketimes[i]))
        return np.argmax(spike_cnt)


if __name__ == '__main__':

    np.random.seed(123456)

    left_data_set, right_data_set = load_eeg_data("data/data2.mat")
    print left_data_set.shape, right_data_set.shape
    dataset = {'left_data_set': left_data_set, 'right_data_set': right_data_set}
    sio.savemat('data_set.mat', dataset)

    rnn_model = MetaRNN()
    rnn_model.load('model_back1.pkl')

    rnn2snn = ConvertRNN2SNN(sim_time=15, rnn_model=rnn_model)
    # print rnn2snn.W.shape       # 6, 50
    # print rnn2snn.W_re.shape    # 100, 100
    # print rnn2snn.W_in.shape    # 50, 100
    # print rnn2snn.W_out.shape   # 100, 2
    print np.max(np.abs(rnn2snn.W_in))
    print np.max(np.abs(rnn2snn.W_re))

    accuracy = []
    err_cnt = 0
    for i in range(left_data_set.shape[0]):

        prected_label = rnn2snn.predictSingleCase(left_data_set[i])
        true_label = 0
        print 'prected label: ', prected_label
        print 'true label: ', true_label
        if prected_label != true_label:
            err_cnt += 1
        rnn2snn.snn_model.end()
    print 'Test set Accuracy: %f%%.' % (100.*(left_data_set.shape[0]-err_cnt)/left_data_set.shape[0],)
    accuracy.append(100.*(left_data_set.shape[0]-err_cnt)/left_data_set.shape[0])

    raw_input("input anything to continue...")

    err_cnt = 0
    for i in range(right_data_set.shape[0]):
        # rnn_model = MetaRNN()
        # rnn_model.load('model.pkl')
        #
        # rnn2snn = ConvertRNN2SNN(sim_time=150, rnn_model=rnn_model)
        # print rnn2snn.W.shape       # 6, 50
        # print rnn2snn.W_re.shape    # 100, 100
        # print rnn2snn.W_in.shape    # 50, 100
        # print rnn2snn.W_out.shape   # 100, 2
        # print np.max(np.abs(rnn2snn.W_in))
        # print np.max(np.abs(rnn2snn.W_re))
        prected_label = rnn2snn.predictSingleCase(right_data_set[i])
        true_label = 1
        print 'prected label: ', prected_label
        print 'true label: ', true_label
        if prected_label != true_label:
            err_cnt += 1
        rnn2snn.snn_model.end()
    print 'Test set Accuracy: %f%%.' % (100.*(right_data_set.shape[0]-err_cnt)/right_data_set.shape[0],)
    accuracy.append(100.*(right_data_set.shape[0]-err_cnt)/right_data_set.shape[0])

    print accuracy

    print 'max hidden neurons spike counts: ', max(rnn2snn.snn_model.hidden_spike_cnts)

