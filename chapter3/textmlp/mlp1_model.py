#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Shutian
# @Date  : 2019-9-17
# @Desc  : dissertation chapter3 mlp model

from __future__ import print_function

import tensorflow as tf

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
    def __init__(self, layer1_dim, layer2_dim, layer3_dim, layer4_dim, x_lsi, x_lda, x_d2v,
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
        self.x_dim1 = x_lsi
        self.x_dim2 = x_lda
        self.x_dim3 = x_d2v
        self.y_dim = y_dim
        self.learning_rate = learning_rate
        self.data_num = data_num

        """
        input data
        """
        # training data: record and label
        self.dropout_keep = tf.placeholder(dtype=tf.float32, name='dropout_keep')
        self.xalsi = tf.placeholder(tf.float32, shape=(None, self.x_dim1), name='xa-lsi')
        self.xalda = tf.placeholder(tf.float32, shape=(None, self.x_dim2), name='xa-lda')
        self.xad2v = tf.placeholder(tf.float32, shape=(None, self.x_dim3), name='xa-d2v')
        self.xblsi = tf.placeholder(tf.float32, shape=(None, self.x_dim1), name='xb-lsi')
        self.xblda = tf.placeholder(tf.float32, shape=(None, self.x_dim2), name='xb-lda')
        self.xbd2v = tf.placeholder(tf.float32, shape=(None, self.x_dim3), name='xb-d2v')
        self.y = tf.placeholder(tf.float32, shape=(None, self.y_dim), name='y-input')

        """
        graph structure
        """
        # predict data: label
        self.y_pred = self.MLP()

        """
        model training 
        """
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.y))
        self.result = tf.nn.softmax(self.y_pred)
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer')
        self.train_op = self.optimizer.minimize(self.loss, name='train_op')

    def MLP(self):
        # network parameter
        with tf.variable_scope("layer1"):
            self.W11 = tf.get_variable("w11", initializer=tf.random_normal([self.x_dim1+self.x_dim2+self.x_dim3, self.layer1_dim], stddev=0.1),
                                      dtype=tf.float32)
            self.W12 = tf.get_variable("w12", initializer=tf.random_normal([self.x_dim1+self.x_dim2+self.x_dim3, self.layer1_dim], stddev=0.1),
                                      dtype=tf.float32)
            self.b11 = tf.get_variable("b11", initializer=tf.zeros([self.layer1_dim]), dtype=tf.float32)
            self.b12 = tf.get_variable("b12", initializer=tf.zeros([self.layer1_dim]), dtype=tf.float32)


        with tf.variable_scope("layer2"):
            self.W2 = tf.get_variable("w2",
                                      initializer=tf.random_normal([self.layer1_dim*2, self.layer2_dim], stddev=0.1),
                                      dtype=tf.float32)
            self.b2 = tf.get_variable("b2", initializer=tf.zeros([self.layer2_dim]), dtype=tf.float32)

        with tf.variable_scope("output"):
            self.W3 = tf.get_variable("w_output",
                                      initializer=tf.truncated_normal([self.layer2_dim, self.y_dim], stddev=0.1),
                                      dtype=tf.float32)
            self.b3 = tf.get_variable("b3", initializer=tf.zeros([self.y_dim]), dtype=tf.float32)


        avec1 = tf.concat([self.xalsi, self.xalda], axis=1)
        avec2 = tf.concat([avec1, self.xad2v], axis=1)
        bvec1 = tf.concat([self.xblsi, self.xblda], axis=1)
        bvec2 = tf.concat([bvec1, self.xbd2v], axis=1)


        # layer 1
        hidden11 = tf.nn.elu(tf.matmul(avec2, self.W11) + self.b11)
        hidden12 = tf.nn.elu(tf.matmul(bvec2, self.W12) + self.b12)

        hidden11_drop = tf.nn.dropout(hidden11, self.dropout_keep)
        hidden12_drop = tf.nn.dropout(hidden12, self.dropout_keep)

        # layer 2
        self.hidden2 = tf.nn.elu(tf.matmul(tf.concat([hidden11_drop, hidden12_drop], axis=1), self.W2) + self.b2)
        hidden2_drop = tf.nn.dropout(self.hidden2, self.dropout_keep)

        # layer 4
        y_pred = tf.matmul(hidden2_drop, self.W3) + self.b3
        return y_pred

    def CNN(self):
        pass



