# -*- coding=utf-8 -*-
# __author__ = 'xxq'

import numpy as np
import scipy.io as sio
import csv

def load_eeg_data(fname='data/DATA.mat'):
    data = sio.loadmat(fname)
    data_x = np.array(data['X_DATA_BASECOR'][0]) # data_x.shape: (550,)
    data_y = np.array(data['Y_DATA_POSTPRE'])[0] # data_y.shape: (550,)
    left, = np.where(data_y == 0)
    right, = np.where(data_y == 1)
    n_seq_l = len(left)             # 186
    print 'length of left: ', n_seq_l
    n_seq_r = len(right)            # 174
    print 'length of right: ', n_seq_r
    n_seq_total = n_seq_l + n_seq_r
    n_steps, n_in = data_x[0].shape # 140, 6
    # print n_seq_l, n_seq_r, n_steps, n_in
    SemanticRecollected = np.zeros((n_seq_l, n_steps, n_in))
    SemanticNonRecollected = np.zeros((n_seq_r, n_steps, n_in))
    for i, sam_i in enumerate(left):
        SemanticRecollected[i, :, :] = data_x[sam_i]
    for i, sam_i in enumerate(right):
        SemanticNonRecollected[i, :, :] = data_x[sam_i]
    # print SemanticRecollected.shape, SemanticNonRecollected.shape
    return SemanticRecollected, SemanticNonRecollected

def spilit_eeg_data(train_ratio=0.6, valid_ratio=0.1, test_ratio=0.3, fname='data/DATA.mat'):
    SemanticRecollected, SemanticNonRecollected = load_eeg_data(fname)
    n_seq_l, n_steps, n_in = SemanticRecollected.shape
    n_seq_r = SemanticNonRecollected.shape[0]

    range_T = range(n_seq_l)
    np.random.shuffle(range_T)
    range_F = range(n_seq_r)
    np.random.shuffle(range_F)
    test_epochs_T = int(np.floor(n_seq_l*test_ratio))
    test_epochs_F = int(np.floor(n_seq_r*test_ratio))
    test_epochs = test_epochs_T + test_epochs_F
    valid_epochs_T = int(np.floor(n_seq_l*valid_ratio))
    valid_epochs_F = int(np.floor(n_seq_r*valid_ratio))
    valid_epochs = valid_epochs_T + valid_epochs_F
    train_epochs_T = n_seq_l - test_epochs_T - valid_epochs_T
    train_epochs_F = n_seq_r - test_epochs_F - valid_epochs_F
    train_epochs = train_epochs_T + train_epochs_F
    # print test_epochs_T, test_epochs_F, test_epochs
    # print valid_epochs_T, valid_epochs_F, valid_epochs
    # print train_epochs_T, train_epochs_F, train_epochs

    test_x = np.zeros((test_epochs, n_steps, n_in)); test_y = np.zeros((test_epochs, n_steps))
    valid_x = np.zeros((valid_epochs, n_steps, n_in)); valid_y = np.zeros((valid_epochs, n_steps))
    train_x = np.zeros((train_epochs, n_steps, n_in)); train_y = np.zeros((train_epochs, n_steps))

    for i in range(test_epochs_T):
        test_x[i, :, :] = SemanticRecollected[range_T[i], :, :]
        test_y[i, :] = 0
    for i in range(test_epochs_F):
        test_x[i+test_epochs_T, :, :] = SemanticNonRecollected[range_F[i], :, :]
        test_y[i+test_epochs_T, :] = 1

    for i in range(valid_epochs_T):
        valid_x[i, :, :] = SemanticRecollected[range_T[i+test_epochs_T], :, :]
        valid_y[i, :] = 0
    for i in range(valid_epochs_F):
        valid_x[i+valid_epochs_T, :, :] = SemanticNonRecollected[range_F[i+test_epochs_F], :, :]
        valid_y[i+valid_epochs_T, :] = 1

    for i in range(train_epochs_T):
        train_x[i, :, :] = SemanticRecollected[range_T[i+test_epochs_T+valid_epochs_T], :, :]
        train_y[i, :] = 0
    for i in range(train_epochs_F):
        train_x[i+train_epochs_T, :, :] = SemanticNonRecollected[range_F[i+test_epochs_F+valid_epochs_F], :, :]
        train_y[i+train_epochs_T, :] = 1

    return train_x, train_y, valid_x, valid_y, test_x, test_y

class WristMovementEEG(object):

    def eeg_data_load(self, filename):
        """
        load eeg sample data from csv file
        :param filename:
        :return:
        """

        csv_file = open(filename, 'r')
        reader = csv.reader(csv_file, delimiter=',')
        data = []
        for line in reader:
            li = [float(i) for i in line]
            data.append(li)
        return data

    def load_true_label(self, filename):
        csv_file = open(filename, 'r')
        reader = csv.reader(csv_file)
        data = []
        for line in reader:
            data.append(int(line[0])-1)
        return data

    def load_data(self):
        data = np.zeros((60, 128, 14))
        num_samples = 60
        for sam_i in range(num_samples):
            filename = './data/wrist_movement_eeg/sam%d_eeg.csv' % (sam_i+1,)
            data_sam_i = np.array(self.eeg_data_load(filename))
            data_sam_i = (data_sam_i-np.mean(data_sam_i, axis=0))/np.std(data_sam_i, axis=0)
            data[sam_i,:,:] = data_sam_i
        # print data.shape

        true_label = self.load_true_label('./data/wrist_movement_eeg/tar_class_labels.csv')
        # print true_label
        return data, true_label

    def spilit_data(self):
        train_x = np.zeros((30, 128, 14)); train_y = np.zeros((30, 128))
        test_x = np.zeros((30, 128, 14)); test_y = np.zeros((30, 128))
        num_samples = 60
        training_samples = range(0,10); training_samples.extend(range(20,30)); training_samples.extend(range(40,50))
        test_samples = range(10,20); test_samples.extend(range(30,40)); test_samples.extend(range(50, 60))
        data, true_label = self.load_data()
        for i, j in enumerate(training_samples):
            train_x[i, :, :] = data[j]
            train_y[i, :] = true_label[j]
        for i, j in enumerate(test_samples):
            test_x[i, :, :] = data[j]
            test_y[i, :] = true_label[j]
        return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    train_x, train_y, valid_x, valid_y, test_x, test_y = spilit_eeg_data(fname="data/DATA10.mat")
    print train_x.shape, train_y.shape
    print valid_x.shape, valid_y.shape
    print test_x.shape, test_y.shape

    # wrist_movement_eeg = WristMovementEEG()
    # train_x, train_y, test_x, test_y = wrist_movement_eeg.spilit_data()
    # print train_x.shape
    # print train_y[:, 0]
    # print test_x.shape
    # print test_y[:, 0]