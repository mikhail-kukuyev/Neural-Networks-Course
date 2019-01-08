import tensorflow as tf

import copy
import codecs
import os
import collections
import numpy as np
import pickle

import argparse


class TextLoader(object):
    def __init__(self, input_filename=None, vocab_filename='vocabulary.pkl'):
        if input_filename is None:
            with open(vocab_filename, 'rb') as f:
                self.chars = pickle.load(f)
                self.init_vocabulary()
        else:
            with codecs.open(input_filename, encoding='utf-8') as f:
                text = f.read()

            counter = collections.Counter(text)
            count_pairs = sorted(counter.items(), key=lambda x: -x[1])
            self.chars, _ = zip(*count_pairs)
            with open(vocab_filename, 'wb') as f:
                pickle.dump(self.chars, f)
            self.init_vocabulary()
            self.tensor = np.array(list(map(self.vocab.get, text)))

    def init_vocabulary(self):
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.vocab_size = len(self.chars)
        self.chars_dict = dict(enumerate(self.chars))

    def text_to_tensor(self, text):
        return np.array(list(map(self.vocab.get, text)))

    def tensor_to_text(self, tensor):
        return ''.join(self.chars_dict[i] for i in tensor)


def batch_generator(tensor, batch_size, seq_len):
    tensor = copy.copy(tensor)
    batch_len = batch_size * seq_len
    num_batches = len(tensor) // batch_len
    tensor = tensor[:batch_len * num_batches]
    tensor = tensor.reshape((batch_size, -1))

    while True:
        np.random.shuffle(tensor)
        for n in range(0, tensor.shape[1], seq_len):
            x = tensor[:, n:n + seq_len]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y


class CharRNN(object):

    @staticmethod
    def build_cell(lstm_size, keep_prob):
        lstm = tf.nn.rnn_cell.LSTMCell(lstm_size)
        drop = tf.nn.rnn_cell.DropoutWrapper(cell=lstm, input_keep_prob=1.0, output_keep_prob=keep_prob,
                                             state_keep_prob=1.0)
        return drop

    def build_lstm(self, vocab_size, lstm_size, num_layers, batch_size=100, seq_len=100, sample=False,
                   input_keep_prob=1.0, output_keep_prob=1.0, state_keep_prob=1.0):
        if sample:
            batch_size = seq_len = 1
            output_keep_prob = 1.0

        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=(batch_size, seq_len), name='inputs')
            self.outputs = tf.placeholder(tf.int32, shape=(batch_size, seq_len), name='outputs')
            self.lstm_inputs = tf.one_hot(self.inputs, vocab_size)

        def build_cell():
            lstm = tf.nn.rnn_cell.LSTMCell(lstm_size)
            if sample or (input_keep_prob == 1.0 and output_keep_prob == 1.0 and state_keep_prob == 1.0):
                return lstm
            else:
                return tf.nn.rnn_cell.DropoutWrapper(cell=lstm, input_keep_prob=input_keep_prob,
                                                     output_keep_prob=output_keep_prob,
                                                     state_keep_prob=state_keep_prob)

        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell([CharRNN.build_cell(lstm_size, output_keep_prob) for _ in range(num_layers)])
            self.initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell=cell, inputs=self.lstm_inputs,
                                                                    initial_state=self.initial_state)

            seq_output = tf.concat(self.lstm_outputs, axis=1)
            x = tf.reshape(seq_output, shape=[-1, lstm_size])

            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal(shape=[lstm_size, vocab_size], stddev=1.0))
                softmax_b = tf.Variable(tf.zeros(vocab_size))

            self.logits = tf.nn.xw_plus_b(x, softmax_w, softmax_b)
            self.prediction = tf.nn.softmax(logits=self.logits, name='predictions')

        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.outputs, vocab_size)
            y_reshaped = tf.reshape(y_one_hot, shape=self.logits.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)


def train(input_file, model_dir, steps, batch_size, seq_len):
    model_path = os.path.join(model_dir, 'model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    text_loader = TextLoader(input_file, os.path.join(model_dir, 'vocabulary.pkl'))
    tensor = text_loader.tensor
    batches = batch_generator(tensor, batch_size, seq_len)

    with tf.Session() as sess:
        char_rnn = CharRNN()
        char_rnn.build_lstm(
            vocab_size=text_loader.vocab_size,
            batch_size=batch_size,
            seq_len=seq_len,
            lstm_size=128,
            num_layers=2,
            input_keep_prob=0.8,
            output_keep_prob=0.6,
            state_keep_prob=0.8)

        tf.summary.scalar('loss', char_rnn.loss)
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join(model_dir, 'summary'))
        writer.add_graph(sess.graph)

        # clipping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(ys=char_rnn.loss, xs=tvars), clip_norm=5)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.apply_gradients(grads_and_vars=zip(grads, tvars))

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        new_state = sess.run(char_rnn.initial_state)

        for step in range(steps):
            x, y = next(batches)
            _, new_state, loss, summ = sess.run([train_op, char_rnn.final_state, char_rnn.loss, summaries],
                                          feed_dict={
                                              char_rnn.inputs: x,
                                              char_rnn.outputs: y,
                                              char_rnn.initial_state: new_state})
            writer.add_summary(summ, step + 1)
            if step % 100 == 0 or step == steps - 1:
                print(f"step: {step}/{steps}, loss: {loss}")
                saver.save(sess, os.path.join(model_path, 'model.ckpt'))


def pick_from_top_n(probs, vocab_size, top_n=5):
    p = np.squeeze(probs)
    p[np.argsort(p)[:-top_n]] = 0
    p /= sum(p)
    return np.random.choice(vocab_size, 1, p=p)[0]


def sample(model_dir, start_string='', max_length=500):
    text_loader = TextLoader(vocab_filename=os.path.join(model_dir, 'vocabulary.pkl'))
    model_path = tf.train.latest_checkpoint(os.path.join(model_dir, 'model'))

    char_rnn = CharRNN()
    char_rnn.build_lstm(
        vocab_size=text_loader.vocab_size,
        lstm_size=128,
        num_layers=2,
        sample=True)

    start = text_loader.text_to_tensor(start_string)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        samples = [c for c in start]
        new_state = sess.run(char_rnn.initial_state)
        probs = np.ones((text_loader.vocab_size,))

        def predict_char(cur_char):
            x = np.zeros((1, 1))
            x[0, 0] = cur_char
            return sess.run([char_rnn.prediction, char_rnn.final_state],
                            feed_dict={char_rnn.inputs: x, char_rnn.initial_state: new_state})

        for c in start:
            probs, new_state = predict_char(c)

        c = pick_from_top_n(probs, text_loader.vocab_size)
        samples.append(c)

        for i in range(max_length):
            probs, new_state = predict_char(c)
            top_n = 30 if text_loader.chars_dict[c].isspace() else 5
            c = pick_from_top_n(probs, text_loader.vocab_size, top_n)
            samples.append(c)

        samples = np.array(samples)
        return text_loader.tensor_to_text(samples)


def parse_args():
    parser = argparse.ArgumentParser(prog='trains char-rnn and samples text using it')
    parser.add_argument('mode', choices=['train', 'sample'],
                        help="program mode, use 'train' for training model, 'sample' for text sampling")
    parser.add_argument('model_dir', help='path to folder where trained model is stored or should be saved,'
                                          ' depending on the specified mode')
    parser.add_argument('--input_file', default='input.txt', help='path to utf-8 encoded input file with data,'
                                                                  'required in train mode')
    parser.add_argument('--start_str', default='', help='use this string to start generating')
    return parser.parse_args()


def main(argv):
    args = parse_args()

    if args.mode == 'train':
        train(args.input_file, args.model_dir, steps=20000, batch_size=50, seq_len=100)
        print(f"Model was successfully trained and saved in '{args.model_dir}' directory.")
    elif args.mode == 'sample':
        text = sample(args.model_dir, start_string=args.start_str, max_length=1000)
        print(text)


if __name__ == "__main__":
    parse_args()
    # tf.logging.set_verbosity(tf.logging.ERROR)
    tf.app.run()
