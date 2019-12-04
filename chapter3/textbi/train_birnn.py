#! /usr/bin/env python
#coding=utf-8

import tensorflow as tf
import numpy as np
import data_input_helper as data_helpers
from text_birnn import TextBiRNN, Config
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# Parameters
# ==================================================
def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--path', default='/space1/google_groups/zheng_shutian/dissertation/chapter3/data/',
                        help='data path')
    parser.add_argument('--result-path', default='/space1/google_groups/zheng_shutian/dissertation/chapter3/result/',
                        help='data path')
    parser.add_argument('--embedding_path',
                        default='/space1/google_groups/zheng_shutian/dissertation/chapter3/pretrain/glove.txt',
                        help='embedding path')
    parser.add_argument('--train', default='aa', help='data path')
    parser.add_argument('--test', default='aa', help='data path')
    parser.add_argument('--embedding_size', type=int, default=300,
                        help="Dimensionality of character embedding (default: 300)")

    parser.add_argument('--hidden_dim', type=int, default=128, help="Number of filters per filter size (default: 128)")
    parser.add_argument('--hidden_dim1', type=int, default=128, help="Dim Number (default: 128)")
    parser.add_argument('--num_layers', type=int, default=2, help="Dropout keep probability (default: 0.5)")
    parser.add_argument('--num_epochs', type=int, default=5, help=' ')
    parser.add_argument('--batch-size', type=int, default=8, help=' ')
    args = parser.parse_args()
    print(args)
    return args


def run(args):
    config = Config(args)

    """Loads starter word-vectors and train/test data."""
    # Load the starter word vectors
    print("Loading data...")
    x1_text, x2_text, y_train = data_helpers.load_data_and_labels(args.path + 'bi-train-'+args.train+'.tsv')
    t1_text, t2_text, y_test = data_helpers.load_data_and_labels(args.path + 'bi-test-'+args.test+'.tsv')

    vocab = data_helpers.load_vocab(x1_text, x2_text, t1_text, t2_text)
    matrix = data_helpers.load_word_embedding_matrix(args.embedding_path, vocab, args.embedding_size)

    w2v_model = matrix['Embedding_matrix']
    shape_word_vocab = matrix['word_vocab']
    max_document_length = 300
    # max_document_length = max(max([len(x.split(" ")) for x in x1_text]),  max([len(x.split(" ")) for x in x2_text]))
    # print('len(x1) = ', len(x1_text), 'len(x2) = ', len(x2_text), ' ', len(y_train))
    print(' max_document_length = ', max_document_length)
    x1_train = data_helpers.get_text_idx(x1_text, shape_word_vocab, max_document_length)
    x2_train = data_helpers.get_text_idx(x2_text, shape_word_vocab, max_document_length)
    x1_test = data_helpers.get_text_idx(t1_text, shape_word_vocab, max_document_length)
    x2_test = data_helpers.get_text_idx(t2_text, shape_word_vocab, max_document_length)
    vocab_size = len(shape_word_vocab)

    """Training."""
    BATCH_SIZE = args.batch_size
    DATA_NUM = len(x1_train)
    STEPS = DATA_NUM // BATCH_SIZE + 1  # STEPS = number of batches

    with tf.Graph().as_default(), tf.Session() as sess:
        birnn = TextBiRNN(
            w2v_model,
            sequence_length=x1_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=vocab_size,
            embedding_size=args.embedding_size,
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            hidden_dim1=args.hidden_dim1)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        min_cross = 6.0

        for i in range(args.num_epochs):
            for j in range(STEPS):
                start = (j * BATCH_SIZE) % DATA_NUM
                end = ((j + 1) * BATCH_SIZE) % DATA_NUM
                if end > start:
                    x1_train1 = x1_train[start:end]
                    x2_train1 = x2_train[start:end]
                    y_train1 = y_train[start:end]
                else:
                    x1_train1 = np.concatenate((x1_train[start:], x1_train[:end]), axis=0)
                    x2_train1 = np.concatenate((x2_train[start:], x2_train[:end]), axis=0)
                    y_train1 = np.concatenate((y_train[start:], y_train[:end]), axis=0)

                sess.run(birnn.train_op,
                         feed_dict={birnn.input_x1: x1_train1, birnn.input_x2: x2_train1, birnn.input_y: y_train1})

                if j%200 == 0:
                    model_loss = sess.run(birnn.loss, feed_dict={birnn.input_x1: x1_test, birnn.input_x2: x2_test,
                                                                 birnn.input_y: y_test})
                    if min_cross > model_loss:
                        min_cross = model_loss
                        print("/epoch_%s-batch_%s-total_cross_entropy_%s" % (i, j, model_loss))
                        output1 = open((args.result_path + 'result-bi-glove-'+args.train+'-'+args.test), 'w')
                        output2 = open((args.result_path + 'vector-bi-glove-'+args.train+'-'+args.test), 'w')
                        y_p, yv = sess.run((birnn.result, birnn.last),
                                           feed_dict={birnn.input_x1: x1_test, birnn.input_x2: x2_test,
                                                      birnn.input_y: y_test})
                        for pred in y_p:
                            output1.write(' '.join(str(v) for v in pred)+'\n')
                        for yvv in yv:
                            output2.write(' '.join(str(vv) for vv in yvv)+'\n')
                        output1.close()
                        output2.close()


if __name__ == "__main__":
    args = parse_args()
    run(args)
