#coding=utf-8


import tensorflow as tf
import numpy as np


class Config(object):
    def __init__(self, args):
        self.EMBEDDING_SIZE = args.embedding_size
        self.NUM_EPOCHS = args.num_epochs
        self.BATCH_SIZE = args.batch_size


class TextBiRNNAtt(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, w2v_model, sequence_length, num_classes, vocab_size,
      embedding_size, num_layers, hidden_dim, hidden_dim1, attn_size):


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
        self.attn_size = attn_size

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
        self.y_pred = self.BiRNNAtt()
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.input_y))
        self.result = tf.nn.softmax(self.y_pred)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='optimizer')
        self.train_op = self.optimizer.minimize(self.loss, name='train_op')

    def BiRNNAtt(self):
        # 定义前向RNN Cell
        with tf.name_scope('fw_rnn'), tf.variable_scope('fw_rnn'):
            lstm_fw_cell_list = [tf.contrib.rnn.LSTMCell(self.hidden_dim) for _ in range(self.num_layers)]
            lstm_fw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list),
                                                           output_keep_prob=1.0)

        # 定义反向RNN Cell
        with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
            lstm_bw_cell_list = [tf.contrib.rnn.LSTMCell(self.hidden_dim) for _ in range(self.num_layers)]
            lstm_bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_bw_cell_list),
                                                           output_keep_prob=1.0)

        # Embedding layer
        with tf.variable_scope("embedding"):
            # embedding initializer
            word_embed = tf.get_variable(name="Word_embedding", shape=[self.vocab_size, self.embedding_size], initializer=tf.constant_initializer(np.array(self.w2v_model)), trainable=False)
            # word_embed = tf.Variable( tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name="word_embeddings")
        embedded_chars1 = tf.nn.embedding_lookup(word_embed, self.input_x1)
        embedded_chars2 = tf.nn.embedding_lookup(word_embed, self.input_x2)

        inputs1 = tf.transpose(embedded_chars1, [1, 0, 2])
        inputs1 = tf.reshape(inputs1, [-1, self.embedding_size])
        inputs1 = tf.split(inputs1, self.sequence_length, 0)

        inputs2 = tf.transpose(embedded_chars2, [1, 0, 2])
        inputs2 = tf.reshape(inputs2, [-1, self.embedding_size])
        inputs2 = tf.split(inputs2, self.sequence_length, 0)

        with tf.name_scope('bi_rnn'), tf.variable_scope('bi_rnn'):
            outputs1, state_fw1, state_bw1 = tf.contrib.rnn.static_bidirectional_rnn(
                lstm_fw_cell_m, lstm_bw_cell_m, inputs1, dtype=tf.float32)
            outputs2, state_fw2, state_bw2 = tf.contrib.rnn.static_bidirectional_rnn(
                lstm_fw_cell_m, lstm_bw_cell_m, inputs2, dtype=tf.float32)

        attention_size = self.attn_size
        with tf.name_scope('attention'), tf.variable_scope('attention'):
            attention_w1 = tf.Variable(tf.truncated_normal([2 * self.hidden_dim, attention_size], stddev=0.1),
                                       name='attention_w1')
            attention_b1 = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b1')
            attention_w2 = tf.Variable(tf.truncated_normal([2 * self.hidden_dim, attention_size], stddev=0.1),
                                       name='attention_w2')
            attention_b2 = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b2')
            u_list1 = []
            u_list2 = []
            for t in range(self.sequence_length):
                u_t1 = tf.tanh(tf.matmul(outputs1[t], attention_w1) + attention_b1)
                u_list1.append(u_t1)
                u_t2 = tf.tanh(tf.matmul(outputs2[t], attention_w2) + attention_b2)
                u_list2.append(u_t2)
            u_w1 = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1), name='attention_uw1')
            u_w2 = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1), name='attention_uw2')
            attn_z1 = []
            attn_z2 = []
            for t in range(self.sequence_length):
                z_t1 = tf.matmul(u_list1[t], u_w1)
                attn_z1.append(z_t1)
                z_t2 = tf.matmul(u_list2[t], u_w2)
                attn_z2.append(z_t2)
            # transform to batch_size * sequence_length
            attn_zconcat1 = tf.concat(attn_z1, axis=1)
            attn_zconcat2 = tf.concat(attn_z2, axis=1)

            alpha1 = tf.nn.softmax(attn_zconcat1)
            alpha2 = tf.nn.softmax(attn_zconcat2)
            # transform to sequence_length * batch_size * 1 , same rank as outputs
            alpha_trans1 = tf.reshape(tf.transpose(alpha1, [1, 0]), [self.sequence_length, -1, 1])
            alpha_trans2 = tf.reshape(tf.transpose(alpha2, [1, 0]), [self.sequence_length, -1, 1])
            final_output1 = tf.reduce_sum(outputs1 * alpha_trans1, 0)
            final_output2 = tf.reduce_sum(outputs2 * alpha_trans2, 0)

            w1 = tf.get_variable("w1", shape=[self.hidden_dim*4, self.hidden_dim1],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.constant(0.1, shape=[self.hidden_dim1]), name="b1")
            self.last = tf.nn.relu(tf.matmul(tf.concat([final_output1, final_output2], axis=1), w1) + b1)


        with tf.name_scope("score"):
            w2 = tf.get_variable("w2", shape=[self.hidden_dim1, self.num_classes],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b2")
            self.scores = tf.matmul(self.last, w2) + b2
        return self.scores




