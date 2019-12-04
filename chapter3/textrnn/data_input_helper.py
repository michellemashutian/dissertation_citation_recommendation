import numpy as np
import re
# import word2vec
# import itertools
# from collections import Counter
# import codecs


def load_word_embedding_matrix(embedding_path, vocab, dim):
    word_vocab = []
    embedding_matrix = []
    word_vocab.extend(['UNK'])
    embedding_matrix.append(np.random.uniform(-1.0, 1.0, (1, dim))[0])
    print('Reading embeddings...')
    with open(embedding_path, 'r') as f:
        for line in f:
            if line.split()[0] in vocab:
                word_vocab.append(line.split()[0])
                embedding_matrix.append([float(i) for i in line.split()[1:]])
    return {'word_vocab': word_vocab, 'Embedding_matrix': np.reshape(embedding_matrix, [-1, dim]).astype(np.float32)}


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def removezero( x, y):
    nozero = np.nonzero(y)
    print('removezero',np.shape(nozero)[-1],len(y))

    if(np.shape(nozero)[-1] == len(y)):
        return np.array(x),np.array(y)

    y = np.array(y)[nozero]
    x = np.array(x)
    x = x[nozero]
    return x, y


def read_file_lines(filename, from_size, line_num):
    i = 0
    text = []
    end_num = from_size + line_num
    for line in open(filename):
        if(i >= from_size):
            text.append(line.strip())
        i += 1
        if i >= end_num:
            return text
    return text


def load_vocab(x1, x2, t1, t2):
    vocab = []
    for sentence in x1:
        for word in sentence.split():
            vocab.append(word.lower())
    for sentence in x2:
        for word in sentence.split():
            vocab.append(word.lower())
    for sentence in t1:
        for word in sentence.split():
            vocab.append(word.lower())
    for sentence in t2:
        for word in sentence.split():
            vocab.append(word.lower())
    vocab = list(set(vocab))
    return vocab


def load_data_and_labels(filepath):
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    one_hot_labels = []
    x1_data = []
    x2_data = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.split('\t')
            if len(parts) != 3:
                continue
            x1_data.append(parts[0])
            x2_data.append(parts[1])
            if parts[2].startswith('0'):
                one_hot_labels.append([0, 1])
            else:
                one_hot_labels.append([1, 0])
    return [x1_data, x2_data, np.array(one_hot_labels)]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            # print('epoch = %d,batch_num = %d,start = %d,end_idx = %d' % (epoch,batch_num,start_index,end_index))
            yield shuffled_data[start_index:end_index]


def get_text_idx(text, shape_word_vocab, max_document_length):
    text_array = np.zeros([len(text), max_document_length], dtype=np.int32)
    symbols = {0: 'UNK'}
    print('int to vocab')
    int_to_vocab = {}
    for index_no, word in enumerate(shape_word_vocab):
        int_to_vocab[index_no] = word
    int_to_vocab.update(symbols)
    vocab_to_int = {word: index_no for index_no, word in int_to_vocab.items()}
    for i, x in enumerate(text):
        words = x.split(" ")
        count = 1
        for j, w in enumerate(words):
            if count > max_document_length:
                break
            else:
                if w in vocab_to_int:
                    text_array[i, j] = vocab_to_int[w]
                else:
                    text_array[i, j] = vocab_to_int['UNK']
            count = count + 1
    return text_array


if __name__ == "__main__":
    x_text, y = load_data_and_labels('F:\\pycharm project\\CitationRec-master\\data\\cutclean_label_corpus10000.txt')
    print (len(x_text))
