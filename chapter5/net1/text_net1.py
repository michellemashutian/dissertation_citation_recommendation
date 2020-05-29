# coding=utf-8


import tensorflow as tf
import numpy as np


class Config(object):
    def __init__(self, args):
        self.EMBEDDING_SIZE = args.embedding_size
        self.NUM_EPOCHS = args.num_epochs
        self.BATCH_SIZE = args.batch_size


class TextNet1(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, w2v_model, sequence_length1, sequence_length2, sequence_length3, layer1_dim, layer2_dim, layer3_dim, num_classes, vocab_size,
            embedding_size, num_layers, hidden_dim, attn_size, filter_sizes, num_filters):

        """
        input parameter
        """
        self.w2v_model = w2v_model
        self.sequence_length1 = sequence_length1
        self.sequence_length2 = sequence_length2
        self.sequence_length3 = sequence_length3
        self.layer1_dim = layer1_dim
        self.layer2_dim = layer2_dim
        self.layer3_dim = layer3_dim
        self.num_classes = num_classes
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
        self.input_l1 = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        '''
        graph
        '''
        self.y_pred = self.BiRNNAtt
        # self.y_pred = self.CNN
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.input_l1))
        self.result = tf.nn.softmax(self.y_pred)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='optimizer')
        self.train_op = self.optimizer.minimize(self.loss, name='train_op')

    @property
    def CNN(self):

        # Embedding layer
        with tf.variable_scope("embedding"):
            # embedding initializer
            word_embed = tf.get_variable(name="Word_embedding", shape=[self.vocab_size, self.embedding_size],
                                              initializer=tf.constant_initializer(np.array(self.w2v_model)), trainable=False)
            embedded_chars1 = tf.nn.embedding_lookup(word_embed, self.input_citation)
            embedded_chars_expanded1 = tf.expand_dims(embedded_chars1, -1)
            embedded_chars2 = tf.nn.embedding_lookup(word_embed, self.input_title)
            embedded_chars_expanded2 = tf.expand_dims(embedded_chars2, -1)
            embedded_chars3 = tf.nn.embedding_lookup(word_embed, self.input_content)
            embedded_chars_expanded3 = tf.expand_dims(embedded_chars3, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs1 = []
        pooled_outputs2 = []
        pooled_outputs3 = []

        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                # 2, 3, 4    300    128
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                w_filter1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), dtype=tf.float32, name="w_filter1")
                b_filter1 = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), dtype=tf.float32, name="b_filter1")
                w_filter2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), dtype=tf.float32, name="w_filter2")
                b_filter2 = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), dtype=tf.float32, name="b_filter2")
                w_filter3 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), dtype=tf.float32, name="w_filter3")
                b_filter3 = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), dtype=tf.float32, name="b_filter3")
                conv1 = tf.nn.conv2d(embedded_chars_expanded1, w_filter1, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                conv2 = tf.nn.conv2d(embedded_chars_expanded2, w_filter2, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                conv3 = tf.nn.conv2d(embedded_chars_expanded3, w_filter3, strides=[1, 1, 1, 1], padding="VALID", name="conv")

                # Apply nonlinearity
                h1 = tf.nn.relu(tf.nn.bias_add(conv1, b_filter1), name="relu_h1")
                h2 = tf.nn.relu(tf.nn.bias_add(conv2, b_filter2), name="relu_h2")
                h3 = tf.nn.relu(tf.nn.bias_add(conv3, b_filter3), name="relu_h3")

                # Maxpooling over the outputs
                pooled1 = tf.nn.max_pool(h1, ksize=[1, self.sequence_length1 - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool1")
                pooled2 = tf.nn.max_pool(h2, ksize=[1, self.sequence_length2 - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool2")
                pooled3 = tf.nn.max_pool(h3, ksize=[1, self.sequence_length3 - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool3")

                pooled_outputs1.append(pooled1)
                pooled_outputs2.append(pooled2)
                pooled_outputs3.append(pooled3)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool1 = tf.concat(pooled_outputs1, 3)
        h_pool2 = tf.concat(pooled_outputs2, 3)
        h_pool3 = tf.concat(pooled_outputs3, 3)
        h_pool_flat1 = tf.reshape(h_pool1, [-1, num_filters_total])
        h_pool_flat2 = tf.reshape(h_pool2, [-1, num_filters_total])
        h_pool_flat3 = tf.reshape(h_pool3, [-1, num_filters_total])

        # Add dropout
        final_output1 = tf.nn.dropout(h_pool_flat1, 1.0)
        final_output2 = tf.nn.dropout(h_pool_flat2, 1.0)
        final_output3 = tf.nn.dropout(h_pool_flat3, 1.0)

        with tf.name_scope("score"):
            layer = tf.concat([final_output2, final_output3], axis=1)
            w1 = tf.get_variable("w1", shape=[num_filters_total * 2, self.layer1_dim], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.constant(0.1, shape=[self.layer1_dim]), name="b1")
            layer1 = tf.matmul(layer, w1) + b1

            layer2 = tf.concat([final_output1, layer1], axis=1)
            w2 = tf.get_variable("w2", shape=[num_filters_total * 2, self.layer2_dim], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.1, shape=[self.layer2_dim]), name="b2")

            layer3 = tf.nn.relu(tf.matmul(layer2, w2) + b2)
            w3 = tf.get_variable("w3", shape=[self.layer2_dim, self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
            scores = tf.matmul(layer3, w3)

        return scores



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
            scores = tf.matmul(layer3, w3)

        return scores
