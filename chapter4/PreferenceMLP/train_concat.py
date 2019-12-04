#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: mashutian
@time: 2019-03-10 13:26
@desc:
"""
from __future__ import print_function
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import preprocessing
from numpy.random import RandomState
import numpy as np

from model_concat import Config, CitationRecNet


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--path', default='E:\\deal\\pubmed\\vector\\',
                        help='data path')
    parser.add_argument('--saved-model',
                        default='E:\\deal\\pubme\vector\\saved_model\\',
                        help='data path')
    parser.add_argument('--layer1-dim', type=int, default=70, help='layer1 dimension')
    parser.add_argument('--layer2-dim', type=int, default=50, help='layer2 dimension')
    parser.add_argument('--layer3-dim', type=int, default=80, help='layer3 dimension')
    parser.add_argument('--layer4-dim', type=int, default=50, help='layer3 dimension')
    parser.add_argument('--learning-rate', type=float, default=0.001, help=' ')
    parser.add_argument('--epoch', type=int, default=60, help=' ')
    parser.add_argument('--batch-size', type=int, default=8, help=' ')
    # parser.add_argument('--batch-norm', dest='is_batch_norm', action='store_true')
    # parser.add_argument('--no-batch-norm', dest='is_batch_norm', action='store_false')

    args = parser.parse_args()
    print(args)
    return args


def run(args):
    config = Config(args)  # get all configurations

    # mnist = input_data.read_data_sets(args.path, one_hot=True)
    # # print(mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels)
    #
    # #load data
    # X_train = mnist.train.images
    # X_test = mnist.test.images
    #
    # Y_train = mnist.train.labels
    # Y_test = mnist.test.labels

    X_train1 = np.loadtxt(args.path + 'mlp-train-vec-content')
    X_train2 = np.loadtxt(args.path + 'mlp-train-vec-node')

    X_test1 = np.loadtxt(args.path + 'mlp-test-vec-content')
    X_test2 = np.loadtxt(args.path + 'mlp-test-vec-node')



    # X_train1 = preprocessing.normalize(X_train1, norm='l2')
    # X_test1 = preprocessing.normalize(X_test1, norm='l2')
    # X_train2 = preprocessing.normalize(X_train2, norm='l2')
    # X_test2 = preprocessing.normalize(X_test2, norm='l2')


    # min_max_scaler = preprocessing.MinMaxScaler()
    # X_train1 = min_max_scaler.fit_transform(X_train1)
    # X_train2 = min_max_scaler.fit_transform(X_train2)
    # X_test1 = min_max_scaler.fit_transform(X_test1)
    # X_test2 = min_max_scaler.fit_transform(X_test2)

    max_abs_scaler = preprocessing.MaxAbsScaler()
    X_train1 = max_abs_scaler.fit_transform(X_train1)
    X_train2 = max_abs_scaler.fit_transform(X_train2)
    X_test1 = max_abs_scaler.fit_transform(X_test1)
    X_test2 = max_abs_scaler.fit_transform(X_test2)

    Y_train = []
    with open(args.path + 'train-label', 'r') as f:
        for line in f:
            result = line.rstrip().split(" ")
            vector = [float(x) for x in result]
            Y_train.append(vector)
    Y_test = []
    with open(args.path + 'test-label', 'r') as f:
        for line in f:
            result = line.rstrip().split(" ")
            vector = [float(x) for x in result]
            Y_test.append(vector)

    Y_train = np.asarray(Y_train)
    Y_test = np.asarray(Y_test)

    print("check data:", type(X_train1), type(Y_train), len(X_train1), len(Y_train), len(X_train1[0]), len(Y_train[0]))

    BATCH_SIZE = args.batch_size
    DATA_NUM = len(Y_train)
    STEPS = DATA_NUM // BATCH_SIZE + 1  # STEPS = number of batches
    x_dim1 = len(X_train1[0])
    x_dim2 = len(X_train2[0])
    y_dim = len(Y_train[0])

    with tf.Graph().as_default(), tf.Session() as sess:
        model = CitationRecNet(config.LAYER1_DIM,
                               config.LAYER2_DIM,
                               config.LAYER3_DIM,
                               config.LAYER4_DIM,
                               x_dim1, x_dim2,
                               y_dim,
                               config.LEARNING_RATE, DATA_NUM)
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        max_acc = 0
        min_cross = 2.0

        for i in range(config.EPOCH):
            for j in range(STEPS):
                start = (j * BATCH_SIZE) % DATA_NUM
                end = ((j + 1) * BATCH_SIZE) % DATA_NUM
                if end > start:
                    train_X1 = X_train1[start:end]
                    train_X2 = X_train2[start:end]
                    train_Y = Y_train[start:end]
                else:
                    train_X1 = np.concatenate((X_train1[start:], X_train1[:end]), axis=0)
                    train_X2 = np.concatenate((X_train2[start:], X_train2[:end]), axis=0)
                    train_Y = np.concatenate((Y_train[start:], Y_train[:end]), axis=0)
                sess.run(model.train_op, feed_dict={model.xa: train_X1, model.xb: train_X2, model.y: train_Y, model.dropout_keep: 1.0})

                if j % 1 == 0:
                    y_pred, total_cross_entropy = sess.run((model.y_pred_softmax, model.loss_metric),
                                                                feed_dict={model.xa: X_test1, model.xb: X_test2, model.y: Y_test,
                                                                           model.dropout_keep: 1.0})
                    if min_cross > total_cross_entropy:
                        min_cross = total_cross_entropy
                        # model_path = args.saved_model+"model-cross.ckpt"
                        print("/epoch_%s-batch_%s-total_cross_entropy_%s" % (i, j, total_cross_entropy))
                        # saver.save(sess, model_path)
                        # output = open((args.path + 'output-filter'), 'w')
                        # y_p = sess.run(model.y_pred_softmax, feed_dict={model.xa: X_test1, model.xb: X_test2, model.dropout_keep: 1.0})
                        # for pred in y_p:
                        #     output.write(' '.join(str(v) for v in pred)+'\n')
                        # output.close()


if __name__ == '__main__':
    args = parse_args()
    run(args)
