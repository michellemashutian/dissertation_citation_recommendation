#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: mashutian
@time: 2019-03-10 13:25
@desc:
"""
from __future__ import print_function
# import sys
# sys.path.append("..") #if you want to import python module from other folders,
# you need to append the system path
import tensorflow as tf
from numpy.random import RandomState
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm  # for batch normalization
import numpy as np
from tensorflow.python.training.moving_averages import assign_moving_average

class Config(object):
    def __init__(self, args):
        self.LAYER1_DIM = args.layer1_dim
        self.LAYER2_DIM = args.layer2_dim
        self.LAYER3_DIM = args.layer3_dim
        self.LAYER4_DIM = args.layer4_dim
        self.LEARNING_RATE = args.learning_rate
        self.EPOCH = args.epoch
        self.BATCH_SIZE = args.batch_size


class CitationRecNet(object):
    def __init__(self, layer1_dim, layer2_dim, layer3_dim, layer4_dim, x_dim1, x_dim2,
                 y_dim, learning_rate, data_num):
        # in order to generate same random sequences
        tf.set_random_seed(1)

        """
        input parameter
        """
        self.layer1_dim = layer1_dim
        self.layer2_dim = layer2_dim
        self.layer3_dim = layer3_dim
        self.layer4_dim = layer4_dim
        self.x_dim1 = x_dim1
        self.x_dim2 = x_dim2
        self.y_dim = y_dim
        self.learning_rate = learning_rate
        self.data_num = data_num

        """
        input data
        """
        # training data: record and label
        self.dropout_keep = tf.placeholder(dtype=tf.float32, name='dropout_keep')
        self.xa = tf.placeholder(tf.float32, shape=(None, self.x_dim1), name='xa-input')
        self.xb = tf.placeholder(tf.float32, shape=(None, self.x_dim2), name='xb-input')
        self.y = tf.placeholder(tf.float32, shape=(None, self.y_dim), name='y-input')

        """
        graph structure
        """
        # predict data: label
        self.y_pred = self.MLP()
        # self.y_pred = tf.clip_by_value(self.MLP(), 1e-10, 1.0)
        self.y_pred_softmax = tf.nn.softmax(self.y_pred)

        """
        model training 
        """
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_pred, labels=self.y))
        self.loss_metric = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_pred, labels=self.y))

        # optimizer
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer')
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                                   decay=0.9, momentum=0.0, epsilon=1e-10, name='optimizer')
        self.train_op = self.optimizer.minimize(self.loss, name='train_op')

    def MLP(self):
        with tf.variable_scope("attention"):
            self.W = tf.get_variable("W", initializer=tf.random_normal([self.x_dim1, self.layer1_dim], stddev=0.1),
                                      dtype=tf.float32)
            self.V = tf.get_variable("V", initializer=tf.random_normal([self.layer1_dim, 1], stddev=0.1),
                                      dtype=tf.float32)
            self.b = tf.get_variable("b", initializer=tf.zeros([self.layer1_dim]), dtype=tf.float32)

        with tf.variable_scope("layer1"):
            self.W1 = tf.get_variable("w1", initializer=tf.random_normal([self.x_dim1, self.layer2_dim], stddev=0.1),
                                      dtype=tf.float32)
            self.b1 = tf.get_variable("b1", initializer=tf.zeros([self.layer2_dim]), dtype=tf.float32)

        with tf.variable_scope("layer2"):
            self.W2 = tf.get_variable("w2",
                                      initializer=tf.random_normal([self.layer2_dim, self.layer3_dim], stddev=0.1),
                                      dtype=tf.float32)
            self.b2 = tf.get_variable("b2", initializer=tf.zeros([self.layer3_dim]), dtype=tf.float32)

        with tf.variable_scope("layer3"):
            self.W3 = tf.get_variable("w3",
                                      initializer=tf.random_normal([self.layer3_dim, self.layer4_dim], stddev=0.1),
                                      dtype=tf.float32)
            self.b3 = tf.get_variable("b3", initializer=tf.zeros([self.layer4_dim]), dtype=tf.float32)

        with tf.variable_scope("output"):
            self.W4 = tf.get_variable("w_output",
                                      initializer=tf.truncated_normal([self.layer4_dim, self.y_dim], stddev=0.1),
                                      dtype=tf.float32)

        """
        attention layer
        """

        weight_a = tf.nn.tanh(tf.matmul((tf.matmul(self.xa, self.W) + self.b), self.V))
        weight_b = tf.nn.tanh(tf.matmul((tf.matmul(self.xb, self.W) + self.b), self.V))
        weight_a = tf.exp(weight_a) / (tf.exp(weight_a) + tf.exp(weight_b))
        weight_b = tf.exp(weight_b) / (tf.exp(weight_a) + tf.exp(weight_b))
        new_a = tf.multiply(weight_a, self.xa)
        new_b = tf.multiply(weight_b, self.xb)

        hidden1 = tf.nn.relu(tf.matmul(new_a+new_b, self.W1) + self.b1)
        hidden1_drop = tf.nn.dropout(hidden1, self.dropout_keep)

        hidden2 = tf.nn.relu(tf.matmul(hidden1_drop, self.W2) + self.b2)
        hidden2_drop = tf.nn.dropout(hidden2, self.dropout_keep)

        hidden3 = tf.nn.relu(tf.matmul(hidden2_drop, self.W3) + self.b3)
        hidden3_drop = tf.nn.dropout(hidden3, self.dropout_keep)

        y_pred = tf.matmul(hidden3_drop, self.W4)
        return y_pred

    def CNN(self):
        pass




