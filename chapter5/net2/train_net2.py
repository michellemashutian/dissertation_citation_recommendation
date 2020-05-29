#! /usr/bin/env python
#coding=utf-8

import tensorflow as tf
import numpy as np
import data_helper_net2 as data_helpers
from text_net2 import TextNet2
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


# Parameters
def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    # parser.add_argument('--path', default='E:\\deal\\pubmed\\chapter5\\process\\', help='data path')
    # parser.add_argument('--result-path', default='E:\\deal\\pubmed\\chapter5\\process\\result', help='data path')
    # parser.add_argument('--embedding_path', default='E:\\deal\\pubmed\\chapter5\\process\\data\\embedding.txt', help='embedding path')

    parser.add_argument('--path', default='/space1/google_groups/zheng_shutian/dissertation/chapter/data/', help='data path')
    parser.add_argument('--result-path', default='/space1/google_groups/zheng_shutian/dissertation/chapter/result/', help='data path')
    parser.add_argument('--embedding_path', default='/space1/google_groups/zheng_shutian/dissertation/chapter/embedding.txt', help='embedding path')

    parser.add_argument('--train', default='2', help='data path')
    parser.add_argument('--test', default='2', help='data path')
    parser.add_argument('--embedding_size', type=int, default=200, help="Dimensionality of character embedding (default: 300)")
    parser.add_argument('--attn_size', type=int, default=128, help="")
    parser.add_argument('--hidden_dim', type=int, default=128, help="")
    parser.add_argument('--layer1_dim', type=int, default=256, help="")
    parser.add_argument('--layer2_dim', type=int, default=256, help="")
    parser.add_argument('--layer3_dim', type=int, default=20, help="")
    parser.add_argument('--num_layers', type=int, default=2, help="")
    parser.add_argument('--num_epochs', type=int, default=4, help=' ')
    parser.add_argument('--batch-size', type=int, default=8, help=' ')
    parser.add_argument('--filter_sizes',  default="2,3,4", help="Comma-separated filter sizes (default: '3,4,5')")
    parser.add_argument('--num_filters', type=int, default=128, help="Number of filters per filter size (default: 128)")
    net_args = parser.parse_args()
    print(net_args)
    return net_args


def run(args):
    """Loads starter word-vectors and train/test data."""
    # Load the starter word vectors
    print("Loading data...")
    x1_citation, x1_title, x1_content, y1_l1, y1_l2 = data_helpers.load_data_and_labels(args.path + 'train-'+args.train+'.tsv')
    x2_citation, x2_title, x2_content, y2_l1, y2_l2 = data_helpers.load_data_and_labels(args.path + 'test-'+args.test+'.tsv')
    vocab = data_helpers.load_vocab([x1_citation, x1_title, x1_content, x2_citation, x2_title, x2_content])
    matrix = data_helpers.load_word_embedding_matrix(args.embedding_path, vocab, args.embedding_size)

    w2v_model = matrix['Embedding_matrix']
    shape_word_vocab = matrix['word_vocab']

    print("Finding max length...")
    max_document_length1 = 60
    max_document_length2 = 25
    max_document_length3 = 400

    print("Convert to idx...")
    x1_citation = data_helpers.get_text_idx(x1_citation, shape_word_vocab, max_document_length1)
    x1_title = data_helpers.get_text_idx(x1_title, shape_word_vocab, max_document_length2)
    x1_content = data_helpers.get_text_idx(x1_content, shape_word_vocab, max_document_length3)
    x2_citation = data_helpers.get_text_idx(x2_citation, shape_word_vocab, max_document_length1)
    x2_title = data_helpers.get_text_idx(x2_title, shape_word_vocab, max_document_length2)
    x2_content = data_helpers.get_text_idx(x2_content, shape_word_vocab, max_document_length3)
    vocab_size = len(shape_word_vocab)

    """Training."""
    BATCH_SIZE = args.batch_size
    DATA_NUM = len(x1_citation)
    STEPS = DATA_NUM // BATCH_SIZE + 1  # STEPS = number of batches

    with tf.Graph().as_default(), tf.Session() as sess:
        birnnatt = TextNet2(
            w2v_model,
            sequence_length1=x1_citation.shape[1],
            sequence_length2=x1_title.shape[1],
            sequence_length3=x1_content.shape[1],
            layer1_dim=args.layer1_dim,
            layer2_dim=args.layer2_dim,
            layer3_dim=args.layer3_dim,
            num_classes=y1_l1.shape[1],
            az_classes=y1_l2.shape[1],
            vocab_size=vocab_size,
            embedding_size=args.embedding_size,
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            attn_size=args.attn_size,
            filter_sizes=list(map(int, args.filter_sizes.split(","))),
            num_filters=args.num_filters)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        min_cross = 6.0

        for i in range(args.num_epochs):
            print(i)
            for j in range(STEPS):
                start = (j * BATCH_SIZE) % DATA_NUM
                end = ((j + 1) * BATCH_SIZE) % DATA_NUM
                if end > start:
                    x1_citation1 = x1_citation[start:end]
                    x1_title1 = x1_title[start:end]
                    x1_content1 = x1_content[start:end]
                    y1_l11 = y1_l1[start:end]
                    y1_l21 = y1_l2[start:end]
                else:
                    x1_citation1 = np.concatenate((x1_citation[start:], x1_citation[:end]), axis=0)
                    x1_title1 = np.concatenate((x1_title[start:], x1_title[:end]), axis=0)
                    x1_content1 = np.concatenate((x1_content[start:], x1_content[:end]), axis=0)
                    y1_l11 = np.concatenate((y1_l1[start:], y1_l1[:end]), axis=0)
                    y1_l21 = np.concatenate((y1_l2[start:], y1_l2[:end]), axis=0)

                sess.run(birnnatt.train_op, feed_dict={birnnatt.input_citation: x1_citation1,
                                                       birnnatt.input_title: x1_title1,
                                                       birnnatt.input_content: x1_content1,
                                                       birnnatt.input_l1: y1_l11,
                                                       birnnatt.input_l2: y1_l21})

                if j % 200 == 0:
                    model_loss1 = sess.run(birnnatt.loss1, feed_dict={birnnatt.input_citation: x2_citation,
                                                                      birnnatt.input_title: x2_title,
                                                                      birnnatt.input_content: x2_content,
                                                                      birnnatt.input_l1: y2_l1,
                                                                      birnnatt.input_l2: y2_l2})
                    if min_cross > model_loss1:
                        min_cross = model_loss1
                        print("/epoch_%s-batch_%s-total_loss_%s" % (i, j, model_loss1))
                        output1 = open((args.result_path + 'net2-result.txt'), 'w')
                        output2 = open((args.result_path + 'net2-az-result.txt'), 'w')
                        y_p, y_az = sess.run((birnnatt.result, birnnatt.result1),
                                             feed_dict={birnnatt.input_citation: x2_citation,
                                                        birnnatt.input_title: x2_title,
                                                        birnnatt.input_content: x2_content,
                                                        birnnatt.input_l1: y2_l1,
                                                        birnnatt.input_l2: y2_l2})
                        for pred in y_p:
                            output1.write(' '.join(str(v) for v in pred) + '\n')
                        output1.close()
                        for pred in y_az:
                            output2.write(' '.join(str(v) for v in pred) + '\n')
                        output2.close()


if __name__ == "__main__":
    argss = parse_args()
    run(argss)
