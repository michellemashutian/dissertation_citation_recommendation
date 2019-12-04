#coding=utf-8


import tensorflow as tf
import numpy as np


class Config(object):
    def __init__(self, args):
        self.EMBEDDING_SIZE = args.embedding_size
        self.FILTER_SIZES = args.filter_sizes
        self.NUM_FILTERS = args.num_filters
        self.DROPOUT_KEEP_PROB = args.dropout_keep_prob
        self.NUM_EPOCHS = args.num_epochs
        self.BATCH_SIZE = args.batch_size


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, w2v_model, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, hidden_dim1):
        tf.set_random_seed(1)

        '''
        input parameter
        '''
        self.w2v_model = w2v_model
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.hidden_dim1 = hidden_dim1

        '''
        input data
        '''

        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        '''
        graph
        '''
        self.y_pred = self.CNN()
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.input_y))
        self.result = tf.nn.softmax(self.y_pred)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='optimizer')
        self.train_op = self.optimizer.minimize(self.loss, name='train_op')

    def CNN(self):

        # Embedding layer
        with tf.variable_scope("embedding"):
            # embedding initializer
            word_embed = tf.get_variable(name="Word_embedding", shape=[self.vocab_size, self.embedding_size], initializer=tf.constant_initializer(np.array(self.w2v_model)), trainable=False)
            # word_embed = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name="word_embeddings")
        embedded_chars1 = tf.nn.embedding_lookup(word_embed, self.input_x1)
        embedded_chars_expanded1 = tf.expand_dims(embedded_chars1, -1)
        embedded_chars2 = tf.nn.embedding_lookup(word_embed, self.input_x2)
        embedded_chars_expanded2 = tf.expand_dims(embedded_chars2, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs1 = []
        pooled_outputs2 = []

        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                # 2, 3, 4    300    128
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                w_filter1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), dtype=tf.float32, name="w1")
                b1 = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), dtype=tf.float32, name="b1")
                w_filter2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), dtype=tf.float32, name="w2")
                b2 = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), dtype=tf.float32, name="b2")
                conv1 = tf.nn.conv2d(embedded_chars_expanded1, w_filter1,
                                     strides=[1, 1, 1, 1], padding="VALID", name="conv")
                conv2 = tf.nn.conv2d(embedded_chars_expanded2, w_filter2,
                                     strides=[1, 1, 1, 1], padding="VALID", name="conv")

                # Apply nonlinearity
                h1 = tf.nn.relu(tf.nn.bias_add(conv1, b1), name="relu")
                h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu")

                # Maxpooling over the outputs
                pooled1 = tf.nn.max_pool(
                    h1,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs1.append(pooled1)

                pooled2 = tf.nn.max_pool(
                    h2,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs2.append(pooled2)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool1 = tf.concat(pooled_outputs1, 3)
        h_pool_flat1 = tf.reshape(h_pool1, [-1, num_filters_total])
        h_pool2 = tf.concat(pooled_outputs2, 3)
        h_pool_flat2 = tf.reshape(h_pool2, [-1, num_filters_total])

        # Add dropout
        h_drop1 = tf.nn.dropout(h_pool_flat1, 1.0)
        h_drop2 = tf.nn.dropout(h_pool_flat2, 1.0)

        ww1 = tf.get_variable("ww1", shape=[num_filters_total*2, self.hidden_dim1],
                              initializer=tf.contrib.layers.xavier_initializer())
        bb1 = tf.Variable(tf.constant(0.1, shape=[self.hidden_dim1]), name="bb1")
        self. h_final = tf.nn.elu(tf.matmul(tf.concat([h_drop1, h_drop2], axis=1), ww1) + bb1)

        with tf.name_scope("output"):
            ww2 = tf.get_variable("ww2", shape=[self.hidden_dim1, self.num_classes],
                                  initializer=tf.contrib.layers.xavier_initializer())
            bb2 = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="bb2")
            self.scores = tf.matmul(self.h_final, ww2) + bb2
        return self.scores




