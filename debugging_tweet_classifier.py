################################################
# intialize glove
################################################
# >pip install glove_python
# >wget http://www.google.com/search?q=glove+word+embeddings+download+those+dope+pre+trained+vectors
# >unzip dope_glove_shit.zip
# >cp cmps242_hw5_config.py.example cmps242_hw5_config.py
# >echo "set GLOVE_LOCATION in cmps242_hw5_config.py to one of those files"
################################################
from cmps242_hw5_config import *
from glove import Glove

glove_data = Glove.load_stanford(GLOVE_LOCATION)

print(glove_data.most_similar('test', number=10))

################################################
# file parsing functions
################################################
from nltk.tokenize import TweetTokenizer
import string, re
import collections
import numpy as np
import glove
from glove.glove_cython import fit_vectors, transform_paragraph

# definitions
HC = "HillaryClinton"
DT = "realDonaldTrump"
NA = "none"
HANDLES = [HC, DT, NA]
HANDLE_MAP = {NA: -1, HC: 0, DT: 1}


# read csv file, return handles and tweets
def parse_tweet_csv(file, file_encoding="utf8"):
    # init
    handles, tweets = [], []

    # read file
    linenr = -1
    with open(file, encoding=file_encoding) as input:
        try:
            for line in input:
                linenr += 1
                if linenr == 0: continue

                # get contents
                line = line.split(",")
                if line[0] in HANDLES:  # label and irst line of tweet
                    handles.append(line[0])
                    tweet = ','.join(line[1:])
                    tweets.append(tweet)
                else:  # second+ line of tweet
                    tweet = tweets.pop()
                    tweet += ','.join(line)
                    tweets.append(tweet)
        except Exception as e:
            print("Exception at line {}: {}".format(linenr, e))
            raise e

    # sanity checks
    assert len(handles) == len(tweets)
    print("Found {} tweets in {} lines".format(len(tweets), linenr + 1))

    # return data
    return handles, tweets


##########################################
### coverting tweet strings to numbers ###

# coverting labels to integers
def int_labels(labels):
    return list(map(lambda x: HANDLE_MAP[x], labels))


# tokenizing
_tokenizer = TweetTokenizer()
_punctuation = set(string.punctuation)


def tokenize(tweet, lowercase=True, strip_urls=True, strip_punctuation=True):
    tokens = _tokenizer.tokenize(tweet)
    if lowercase: tokens = list(map(lambda x: x.lower(), tokens))
    if strip_urls: tokens = list(filter(lambda x: not x.startswith("http"), tokens))
    if strip_punctuation:  # https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
        tokens = list(filter(
            lambda x: x.startswith(u'@') or x.startswith(u'#') or x not in _punctuation and not re.match(
                u"[^\w\d'\s$]+", x), tokens))
    return tokens


# glove information
def get_glove_vector(glove_data, tokens, epochs=50, ignore_missing=True):
    """
    tpesout: This code came from the 'glove' repo I'm using (but had a bug, so I needed to slighly modify it)
    https://github.com/maciejkula/glove-python/blob/master/glove/glove.py

    Transform an iterable of tokens into its vector representation
    (a paragraph vector).

    Experimental. This will return something close to a tf-idf
    weighted average of constituent token vectors by fitting
    rare words (with low word bias values) more closely.
    """

    if glove_data.word_vectors is None:
        raise Exception('Model must be fit to transform paragraphs')

    if glove_data.dictionary is None:
        raise Exception('Dictionary must be provided to '
                        'transform paragraphs')

    cooccurrence = collections.defaultdict(lambda: 0.0)

    for token in tokens:
        try:
            cooccurrence[glove_data.dictionary[token]] += glove_data.max_count / 10.0
        except KeyError:
            if not ignore_missing:
                raise

    random_state = glove.glove.check_random_state(glove_data.random_state)

    word_ids = np.array(list(cooccurrence.keys()), dtype=np.int32)
    values = np.array(list(cooccurrence.values()), dtype=np.float64)
    shuffle_indices = np.arange(len(word_ids), dtype=np.int32)

    # Initialize the vector to mean of constituent word vectors
    paragraph_vector = np.mean(glove_data.word_vectors[word_ids], axis=0)
    sum_gradients = np.ones_like(paragraph_vector)

    # Shuffle the coocurrence matrix
    random_state.shuffle(shuffle_indices)
    transform_paragraph(glove_data.word_vectors,
                        glove_data.word_biases,
                        paragraph_vector,
                        sum_gradients,
                        word_ids,
                        values,
                        shuffle_indices,
                        glove_data.learning_rate,
                        glove_data.max_count,
                        glove_data.alpha,
                        epochs)

    return paragraph_vector


# get all tweets
def import_text(tweets):
    return [get_glove_vector(glove_data, tokenize(tweet)) for tweet in tweets]



################################################
# get raw test data
################################################
import random

# init
TEST_RATIO = 0.1
assert TEST_RATIO > 0 and TEST_RATIO < 1

# get data
text_handles, raw_tweets = parse_tweet_csv("train.csv")
handles = int_labels(text_handles)
tweets = import_text(raw_tweets)
data_vector_size = len(tweets[0])

### validation
for i in range(1):
    tweet = raw_tweets[random.randint(0, len(raw_tweets))]
    print(tokenize(tweet))
    print(get_glove_vector(glove_data, tokenize(tweet)))
    print()
# for handle in int_labels(handles[0:7]):
#     print(handle)



################################################
# split test data into train and test
################################################
import pandas as pd

LABEL = 'handle'
DATA = 'tweet_data'
LENGTH = 'length'

# this is so the glove output is in integer form (maybe not necessary)
FLOAT_GRANULARITY = (1 << 16)
VOCAB_SIZE = 2 * FLOAT_GRANULARITY + 1 # for positive and negative

# this is so we can define a range for the values
value_max = 0
for tweet in tweets:
    for x in tweet:
        if abs(x) > value_max: value_max = abs(x)

# split into test and train
train_labels, train_data, test_labels, test_data = list(), list(), list(), list()
for handle, tweet in zip(handles, tweets):
    if np.isnan(tweet[0]): continue #a row of all nan's happens with data that glove can't understand (like, all hashtags)
    tweet = list(map(lambda x: int(x / value_max * FLOAT_GRANULARITY + FLOAT_GRANULARITY), tweet))
    if random.random() < TEST_RATIO:
        test_labels.append(handle)
        test_data.append(tweet)
    else:
        train_labels.append(handle)
        train_data.append(tweet)

# document and validate
print("Separated into {} train and {} test ({}%)\n".format(len(train_data), len(test_data),
                                                         int(100.0 * len(test_data) / len(raw_tweets))))
assert len(train_labels) == len(train_data) and len(train_data) > 0
assert len(test_labels) == len(test_data) and len(test_data) > 0
assert len(test_labels) > len(tweets) * (TEST_RATIO - .05)
assert len(test_labels) < len(tweets) * (TEST_RATIO + .05)

# save to dataframe
train = pd.DataFrame({
    LABEL: train_labels,
    DATA: train_data,
    LENGTH: [data_vector_size for _ in range(len(train_data))]
})
test = pd.DataFrame({
    LABEL: test_labels,
    DATA: test_data,
    LENGTH: [data_vector_size for _ in range(len(test_data))]
})
print(train.head())

################################################
# initializing our tensor
#
# based off of blogpost david parks showed us:
# https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
################################################
import tensorflow as tf


class DataIterator():
    def __init__(self, df):
        self.df = df
        self.size = len(self.df)
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.cursor = 0

    def next_batch(self, n):
        if self.cursor + n - 1 > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df.ix[self.cursor:self.cursor + n - 1]
        start_idx = self.cursor
        self.cursor += n
        # return res['as_numbers'], res['gender']*3 + res['age_bracket'], res['length']
        # return res[DATA], res[LABEL], res[LENGTH]
        data = res[DATA]
        labels = res[LABEL]
        length = res[LENGTH]
        return np.asarray([data[i] for i in range(start_idx, start_idx + len(data))]), \
               np.asarray([labels[i] for i in range(start_idx, start_idx + len(labels))]), \
               np.asarray([length[i] for i in range(start_idx, start_idx + len(length))])


def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()


def build_graph(vocab_size=VOCAB_SIZE, state_size=64, batch_size=256, num_classes=2):
    reset_graph()

    # Placeholders
    x = tf.placeholder(tf.int32, [batch_size, None])  # [batch_size, num_steps]
    seqlen = tf.placeholder(tf.int32, [batch_size])
    y = tf.placeholder(tf.int32, [batch_size])
    keep_prob = tf.placeholder_with_default(1.0, [])

    # Embedding layer
    embeddings = tf.get_variable('embedding_matrix', [vocab_size, state_size])
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    # RNN
    cell = tf.nn.rnn_cell.GRUCell(state_size)
    init_state = tf.get_variable('init_state', [1, state_size],
                                 initializer=tf.constant_initializer(0.0))
    init_state = tf.tile(init_state, [batch_size, 1])
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=seqlen,
                                                 initial_state=init_state)

    # Add dropout, as the model otherwise quickly overfits
    rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)
    idx = tf.range(batch_size) * tf.shape(rnn_outputs)[1] + (seqlen - 1)
    last_rnn_output = tf.gather(tf.reshape(rnn_outputs, [-1, state_size]), idx)

    # Softmax layer
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(last_rnn_output, W) + b
    preds = tf.nn.softmax(logits)
    correct = tf.equal(tf.cast(tf.argmax(preds, 1), tf.int32), y)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    return {
        'x': x,
        'seqlen': seqlen,
        'y': y,
        'dropout': keep_prob,
        'loss': loss,
        'ts': train_step,
        'preds': preds,
        'accuracy': accuracy
    }


def train_graph(g, batch_size=256, num_epochs=10, iterator=DataIterator):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tr = iterator(train)
        te = iterator(test)

        step, accuracy = 0, 0
        tr_losses, te_losses = [], []
        current_epoch = 0
        while current_epoch < num_epochs:
            step += 1
            batch = tr.next_batch(batch_size)
            feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2], g['dropout']: 0.6}
            accuracy_, _ = sess.run([g['accuracy'], g['ts']], feed_dict=feed)
            accuracy += accuracy_

            if tr.epochs > current_epoch:
                current_epoch += 1
                tr_losses.append(accuracy / step)
                step, accuracy = 0, 0

                # eval test set
                te_epoch = te.epochs
                while te.epochs == te_epoch:
                    step += 1
                    batch = te.next_batch(batch_size)
                    feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2]}
                    accuracy_ = sess.run([g['accuracy']], feed_dict=feed)[0]
                    accuracy += accuracy_

                te_losses.append(accuracy / step)
                step, accuracy = 0, 0
                print("Accuracy after epoch", current_epoch, " - tr:", tr_losses[-1], "- te:", te_losses[-1])

    return tr_losses, te_losses



################################################
# run it!
################################################

# this fails, just like us
g = build_graph()
tr_losses, te_losses = train_graph(g)