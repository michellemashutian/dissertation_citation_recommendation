#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Shutian
# @Date  : 2019-9-17
# @Desc  : dissertation chapter3 mlp model

from __future__ import print_function
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import tensorflow as tf
import numpy as np

from mlp1_model import Config, CitationRecNet


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--path', default='/space1/google_groups/zheng_shutian/dissertation/chapter3/data/',
                        help='data path')
    parser.add_argument('--result-path', default='/space1/google_groups/zheng_shutian/dissertation/chapter3/result/',
                        help='data path')
    parser.add_argument('--train', default='aa', help='data path')
    parser.add_argument('--test', default='aa', help='data path')
    parser.add_argument('--layer1-dim', type=int, default=200, help='layer1 dimension')
    parser.add_argument('--layer2-dim', type=int, default=50, help='layer2 dimension')
    parser.add_argument('--layer3-dim', type=int, default=50, help='layer3 dimension')
    parser.add_argument('--layer4-dim', type=int, default=10, help='layer4 dimension')
    parser.add_argument('--learning-rate', type=float, default=0.001, help=' ')
    parser.add_argument('--epoch', type=int, default=5, help=' ')
    parser.add_argument('--batch-size', type=int, default=8, help=' ')

    args = parser.parse_args()
    print(args)
    return args


def run(args):
    config = Config(args)  # get all configurations

    train_alsi = np.loadtxt(args.path + 'train-alsi-'+args.train)
    train_alda = np.loadtxt(args.path + 'train-alda-'+args.train)
    train_ad2v = np.loadtxt(args.path + 'train-ad2v-'+args.train)
    train_blsi = np.loadtxt(args.path + 'train-blsi-'+args.train)
    train_blda = np.loadtxt(args.path + 'train-blda-'+args.train)
    train_bd2v = np.loadtxt(args.path + 'train-bd2v-'+args.train)
    train_label = np.loadtxt(args.path + 'train-label-'+args.train)

    test_alsi = np.loadtxt(args.path + 'test-alsi-'+args.test)
    test_alda = np.loadtxt(args.path + 'test-alda-'+args.test)
    test_ad2v = np.loadtxt(args.path + 'test-ad2v-'+args.test)
    test_blsi = np.loadtxt(args.path + 'test-blsi-'+args.test)
    test_blda = np.loadtxt(args.path + 'test-blda-'+args.test)
    test_bd2v = np.loadtxt(args.path + 'test-bd2v-'+args.test)
    test_label = np.loadtxt(args.path + 'test-label-'+args.test)

    BATCH_SIZE = args.batch_size
    DATA_NUM = len(train_label)
    STEPS = DATA_NUM // BATCH_SIZE + 1  # STEPS = number of batches
    x_lsi = len(train_alsi[0])
    x_lda = len(train_alda[0])
    x_d2v = len(train_ad2v[0])
    y_dim = len(train_label[0])
    print(x_lsi, x_lda, x_d2v)
    print(DATA_NUM, STEPS)

    with tf.Graph().as_default(), tf.Session() as sess:
        model = CitationRecNet(config.LAYER1_DIM,
                               config.LAYER2_DIM,
                               config.LAYER3_DIM,
                               config.LAYER4_DIM,
                               x_lsi, x_lda, x_d2v,
                               y_dim,
                               config.LEARNING_RATE, DATA_NUM)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        min_cross = 6.0

        for i in range(config.EPOCH):
            for j in range(STEPS):
                start = (j * BATCH_SIZE) % DATA_NUM
                end = ((j + 1) * BATCH_SIZE) % DATA_NUM
                if end > start:
                    train_alsi1 = train_alsi[start:end]
                    train_alda1 = train_alda[start:end]
                    train_ad2v1 = train_ad2v[start:end]
                    train_blsi1 = train_blsi[start:end]
                    train_blda1 = train_blda[start:end]
                    train_bd2v1 = train_bd2v[start:end]
                    train_label1 = train_label[start:end]

                else:
                    train_alsi1 = np.concatenate((train_alsi[start:], train_alsi[:end]), axis=0)
                    train_alda1 = np.concatenate((train_alda[start:], train_alda[:end]), axis=0)
                    train_ad2v1 = np.concatenate((train_ad2v[start:], train_ad2v[:end]), axis=0)
                    train_blsi1 = np.concatenate((train_blsi[start:], train_blsi[:end]), axis=0)
                    train_blda1 = np.concatenate((train_blda[start:], train_blda[:end]), axis=0)
                    train_bd2v1 = np.concatenate((train_bd2v[start:], train_bd2v[:end]), axis=0)
                    train_label1 = np.concatenate((train_label[start:], train_label[:end]), axis=0)
                sess.run(model.train_op, feed_dict={model.xalsi: train_alsi1, model.xalda: train_alda1,
                                                    model.xad2v: train_ad2v1,
                                                    model.xblsi: train_blsi1, model.xblda: train_blda1,
                                                    model.xbd2v: train_bd2v1,
                                                    model.y: train_label1, model.dropout_keep: 1.0})

                if i == 4:
                    model_loss = sess.run(model.loss,
                                          feed_dict={model.xalsi: test_alsi, model.xalda: test_alda,
                                                     model.xad2v: test_ad2v,
                                                     model.xblsi: test_blsi, model.xblda: test_blda,
                                                     model.xbd2v: test_bd2v,
                                                     model.y: test_label, model.dropout_keep: 1.0})
                    if min_cross > model_loss:
                        min_cross = model_loss
                        print("/epoch_%s-batch_%s-total_cross_entropy_%s" % (i, j, model_loss))
                        output1 = open((args.result_path + 'result-mlp1-'+args.train+'-'+args.test), 'w')
                        output2 = open((args.result_path + 'vector-mlp1-'+args.train+'-'+args.test), 'w')
                        y_p, yv = sess.run((model.result, model.hidden2),
                                           feed_dict={model.xalsi: test_alsi, model.xalda: test_alda,
                                                      model.xad2v: test_ad2v,
                                                      model.xblsi: test_blsi, model.xblda: test_blda,
                                                      model.xbd2v: test_bd2v,
                                                      model.y: test_label, model.dropout_keep: 1.0})
                        for pred in y_p:
                            output1.write(' '.join(str(v) for v in pred)+'\n')
                        for yvv in yv:
                            output2.write(' '.join(str(vv) for vv in yvv)+'\n')
                        output1.close()
                        output2.close()



if __name__ == '__main__':
    args = parse_args()
    run(args)
