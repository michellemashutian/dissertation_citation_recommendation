# coding=utf-8


import tensorflow as tf
import numpy as np


class Config(object):
    def __init__(self, args):
        self.EMBEDDING_SIZE = args.embedding_size
        self.NUM_EPOCHS = args.num_epochs
        self.BATCH_SIZE = args.batch_size


class TextNet6(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, w2v_model, sequence_length1, sequence_length2, sequence_length3, layer1_dim,
            layer2_dim, layer3_dim,
            num_classes, az_classes, time_slices, vocab_size,
            embedding_size, num_layers, hidden_dim, attn_size, filter_sizes, num_filters):

        '''
        input parameter
        '''
        self.w2v_model = w2v_model
        self.sequence_length1 = sequence_length1
        self.sequence_length2 = sequence_length2
        self.sequence_length3 = sequence_length3
        self.layer1_dim = layer1_dim
        self.layer2_dim = layer2_dim
        self.layer3_dim = layer3_dim
        self.num_classes = num_classes
        self.az_classes = az_classes
        self.time_slices = time_slices
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.attn_size = attn_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        '''
        input data
        '''

        # Placeholders for input, output and dropout
        self.input_citation = tf.placeholder(tf.int32, [None, sequence_length1], name="input_citation")
        self.input_title = tf.placeholder(tf.int32, [None, sequence_length2], name="input_title")
        self.input_content = tf.placeholder(tf.int32, [None, sequence_length3], name="input_content")
        self.input_tp = tf.placeholder(tf.float32, [None, 2], name="input_tp")
        self.input_l1 = tf.placeholder(tf.float32, [None, num_classes], name="input_y1")
        self.input_l2 = tf.placeholder(tf.float32, [None, az_classes], name="input_y2")

        '''
        graph
        '''

        self.y_pred, self.y_az = self.BiRNNAtt
        self.loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.input_l1))
        self.loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_az, labels=self.input_l2))
        self.loss = self.loss1+self.loss2 * 0.2
        self.result = tf.nn.softmax(self.y_pred)
        self.result1 = tf.nn.softmax(self.y_az)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='optimizer')
        self.train_op = self.optimizer.minimize(self.loss, name='train_op')


    @property
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
            word_embed = tf.get_variable(name="Word_embedding", shape=[self.vocab_size, self.embedding_size],
                                         initializer=tf.constant_initializer(np.array(self.w2v_model)),
                                         trainable=False)
            embedded_chars1 = tf.nn.embedding_lookup(word_embed, self.input_citation)
            embedded_chars2 = tf.nn.embedding_lookup(word_embed, self.input_title)
            embedded_chars3 = tf.nn.embedding_lookup(word_embed, self.input_content)

            inputs1 = tf.transpose(embedded_chars1, [1, 0, 2])
            # 转换成(batch_size * sequence_length, rnn_size)
            inputs1 = tf.reshape(inputs1, [-1, self.embedding_size])
            # 转换成list,里面的每个元素是(batch_size, rnn_size)
            inputs1 = tf.split(inputs1, self.sequence_length1, 0)

            inputs2 = tf.transpose(embedded_chars2, [1, 0, 2])
            inputs2 = tf.reshape(inputs2, [-1, self.embedding_size])
            inputs2 = tf.split(inputs2, self.sequence_length2, 0)

            inputs3 = tf.transpose(embedded_chars3, [1, 0, 2])
            inputs3 = tf.reshape(inputs3, [-1, self.embedding_size])
            inputs3 = tf.split(inputs3, self.sequence_length3, 0)

        with tf.name_scope('bi_rnn'), tf.variable_scope('bi_rnn'):
            outputs1, state_fw1, state_bw1 = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell_m,
                                                                                     lstm_bw_cell_m, inputs1,
                                                                                     dtype=tf.float32)
            outputs2, state_fw2, state_bw2 = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell_m,
                                                                                     lstm_bw_cell_m, inputs2,
                                                                                     dtype=tf.float32)
            outputs3, state_fw3, state_bw3 = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell_m,
                                                                                     lstm_bw_cell_m, inputs3,
                                                                                     dtype=tf.float32)

        attention_size = self.attn_size
        with tf.name_scope('attention1'), tf.variable_scope('attention1'):
            attention_w1 = tf.Variable(tf.truncated_normal([2 * self.hidden_dim, attention_size], stddev=0.1),
                                       name='attention_w1')
            attention_b1 = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b1')
            u_list1 = []
            for t in range(self.sequence_length1):
                u_t1 = tf.tanh(tf.matmul(outputs1[t], attention_w1) + attention_b1)
                u_list1.append(u_t1)
            u_w1 = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1), name='attention_uw1')
            attn_z1 = []
            for t in range(self.sequence_length1):
                z_t1 = tf.matmul(u_list1[t], u_w1)
                attn_z1.append(z_t1)
            attn_zconcat1 = tf.concat(attn_z1, axis=1)
            alpha1 = tf.nn.softmax(attn_zconcat1)
            alpha_trans1 = tf.reshape(tf.transpose(alpha1, [1, 0]), [self.sequence_length1, -1, 1])
            final_output1 = tf.reduce_sum(outputs1 * alpha_trans1, 0)

        with tf.name_scope('attention2'), tf.variable_scope('attention2'):
            attention_w2 = tf.Variable(tf.truncated_normal([2 * self.hidden_dim, attention_size], stddev=0.1),
                                       name='attention_w2')
            attention_b2 = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b2')
            u_list2 = []
            for t in range(self.sequence_length2):
                u_t2 = tf.tanh(tf.matmul(outputs2[t], attention_w2) + attention_b2)
                u_list2.append(u_t2)
            u_w2 = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1), name='attention_uw2')
            attn_z2 = []
            for t in range(self.sequence_length2):
                z_t2 = tf.matmul(u_list2[t], u_w2)
                attn_z2.append(z_t2)
            attn_zconcat2 = tf.concat(attn_z2, axis=1)
            alpha2 = tf.nn.softmax(attn_zconcat2)
            alpha_trans2 = tf.reshape(tf.transpose(alpha2, [1, 0]), [self.sequence_length2, -1, 1])
            final_output2 = tf.reduce_sum(outputs2 * alpha_trans2, 0)

        with tf.name_scope('attention3'), tf.variable_scope('attention3'):
            attention_w3 = tf.Variable(tf.truncated_normal([2 * self.hidden_dim, attention_size], stddev=0.1),
                                       name='attention_w3')
            attention_b3 = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b3')
            u_list3 = []
            for t in range(self.sequence_length3):
                u_t3 = tf.tanh(tf.matmul(outputs3[t], attention_w3) + attention_b3)
                u_list3.append(u_t3)
            u_w3 = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1), name='attention_uw3')
            attn_z3 = []
            for t in range(self.sequence_length3):
                z_t3 = tf.matmul(u_list3[t], u_w3)
                attn_z3.append(z_t3)
            attn_zconcat3 = tf.concat(attn_z3, axis=1)
            alpha3 = tf.nn.softmax(attn_zconcat3)
            alpha_trans3 = tf.reshape(tf.transpose(alpha3, [1, 0]), [self.sequence_length3, -1, 1])
            final_output3 = tf.reduce_sum(outputs3 * alpha_trans3, 0)

        with tf.name_scope("score"):
            layer = tf.concat([final_output2, final_output3], axis=1)
            w1 = tf.get_variable("w1", shape=[self.hidden_dim * 4, self.layer1_dim], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.constant(0.1, shape=[self.layer1_dim]), name="b1")
            layer1 = tf.nn.relu(tf.matmul(layer, w1) + b1)
            layer2 = tf.concat([final_output1, layer1], axis=1)

            w2 = tf.get_variable("w2", shape=[self.hidden_dim * 4, self.layer2_dim], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.1, shape=[self.layer2_dim]), name="b2")
            layer3 = tf.nn.relu(tf.matmul(layer2, w2) + b2)

            w3 = tf.get_variable("w3", shape=[self.layer2_dim, self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b3")
            score = tf.matmul(layer3, w3) + b3
            scores = score * self.input_tp

            w4 = tf.get_variable("w4", shape=[self.num_classes, self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b4")
            final_score = tf.matmul(scores, w4) + b4

            w_az = tf.get_variable("w_az", shape=[self.hidden_dim * 2, self.az_classes], initializer=tf.contrib.layers.xavier_initializer())
            b_az = tf.Variable(tf.constant(0.1, shape=[self.az_classes]), name="b_az")
            az_scores = tf.matmul(final_output1, w_az) + b_az

        return final_score, az_scores
