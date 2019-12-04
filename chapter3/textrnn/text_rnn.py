#coding=utf-8


import tensorflow as tf
import numpy as np


class Config(object):
    def __init__(self, args):
        self.EMBEDDING_SIZE = args.embedding_size
        self.NUM_EPOCHS = args.num_epochs
        self.BATCH_SIZE = args.batch_size


class TextRNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, w2v_model, sequence_length, num_classes, vocab_size,
      embedding_size, num_layers, hidden_dim, hidden_dim1, rnn):
        tf.set_random_seed(1)

        '''
        input parameter
        '''
        self.w2v_model = w2v_model
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.hidden_dim1 = hidden_dim1
        self.rnn = rnn

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
        self.y_pred = self.RNN()
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.input_y))
        self.result = tf.nn.softmax(self.y_pred)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='optimizer')
        self.train_op = self.optimizer.minimize(self.loss, name='train_op')

    def RNN(self):
        def lstm_cell():
            # lstm
            return tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=True)

        def gru_cell():
            # gru
            return tf.contrib.rnn.GRUCell(self.hidden_dim)

        def dropout():
            if self.rnn == 'lstm':
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0)

        # Embedding layer
        with tf.variable_scope("embedding"):
            # embedding initializer
            word_embed = tf.get_variable(name="Word_embedding", shape=[self.vocab_size, self.embedding_size], initializer=tf.constant_initializer(np.array(self.w2v_model)), trainable=False)
            # word_embed = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name="word_embeddings")
        embedded_chars1 = tf.nn.embedding_lookup(word_embed, self.input_x1)
        embedded_chars2 = tf.nn.embedding_lookup(word_embed, self.input_x2)

        with tf.name_scope("rnn"):
            # 多层rnn网络
            cells = [dropout() for _ in range(self.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            _outputs1, _state1 = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedded_chars1, dtype=tf.float32)
            last1 = _outputs1[:, -1, :]
            _outputs2, _state2 = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedded_chars2, dtype=tf.float32)
            last2 = _outputs2[:, -1, :]
            ww1 = tf.get_variable("ww1", shape=[self.hidden_dim*2, self.hidden_dim1],
                                  initializer=tf.contrib.layers.xavier_initializer())
            bb1 = tf.Variable(tf.constant(0.1, shape=[self.hidden_dim1]), name="bb1")
            self.last = tf.nn.relu(tf.matmul(tf.concat([last1, last2], axis=1), ww1) + bb1)

        with tf.name_scope("score"):
            w2 = tf.get_variable("w2", shape=[self.hidden_dim1, self.num_classes],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b2")
            self.scores = tf.matmul(self.last, w2) + b2
        return self.scores




