#!/usr/bin/env python3
import pickle
from collections import namedtuple
import tensorflow as tf
import numpy as np

NUM_CLASSES = 10


def to_categorical(arr):
    return np.eye(NUM_CLASSES)[arr]


def retrieve_data(sample):
    x = sample.images
    y = to_categorical(sample.labels)
    return x, y


mnist = tf.contrib.learn.datasets.load_dataset("mnist")
num_features = mnist.train.images.shape[1]

with tf.name_scope('data'):
    x = tf.placeholder(name='x', dtype=tf.float32, shape=(None, num_features))
    y = tf.placeholder(name='y', dtype=tf.float32, shape=(None, NUM_CLASSES))

with tf.name_scope('params'):
    w = tf.Variable(name='W', dtype=tf.float32, initial_value=tf.zeros([num_features, NUM_CLASSES]))
    b = tf.Variable(name='B', dtype=tf.float32, initial_value=tf.zeros([NUM_CLASSES]))

with tf.name_scope('predict_operation'):
    predict_op = tf.nn.softmax(tf.matmul(x, w) + b)

with tf.name_scope('optimization'):
    learning_rate = tf.placeholder(name='learning_rate', dtype=tf.float32)
    avg_loss = tf.placeholder(name='learning_rate', dtype=tf.float32)

    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(predict_op), reduction_indices=1))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

with tf.name_scope('evaluation'):
    prediction = tf.equal(tf.argmax(predict_op, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

init_op = tf.global_variables_initializer()

Record = namedtuple('Record', ['learning_rate', 'batch_size', 'epochs', 'accuracy'])


def validation(rate, batch_size, epochs):
    tf.summary.scalar('cross-entropy', avg_loss)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(f'summary/{rate}/{batch_size}')

    valid_x, valid_y = retrieve_data(mnist.validation)
    data = []

    steps = mnist.train.num_examples // batch_size

    with tf.Session() as session:
        session.run(init_op)
        for epoch in range(epochs):
            avg_loss_val = 0.
            for step in range(steps):
                vx, vy = mnist.train.next_batch(batch_size)
                _, loss_val = session.run([train_op, loss],
                                          feed_dict={x: vx, y: to_categorical(vy), learning_rate: rate})
                avg_loss_val += loss_val

            avg_loss_val /= steps
            acc, summary = session.run([accuracy, merged_summary],
                                       feed_dict={x: valid_x, y: valid_y, avg_loss: avg_loss_val})
            summary_writer.add_summary(summary, global_step=epoch + 1)

            data.append(Record(rate, batch_size, epoch + 1, acc))
            print(f"Learning_rate: {rate} batch_size: {batch_size} epoch: {epoch + 1} accuracy: {acc}")

        summary_writer.add_graph(session.graph)

    summary_writer.flush()
    return data


def search_parameters(learning_rate_grid, batch_size_grid, epochs):
    data = []
    for rate in learning_rate_grid:
        for batch_size in batch_size_grid:
            data += validation(rate, batch_size, epochs)

    pickle.dump(data, open('parameters.bin', 'wb'))


def choose_best_parameters():
    data = pickle.load(open('parameters.bin', 'rb'))
    data = filter(lambda el: el.epochs % 50 == 0, data)
    return max(data, key=lambda el: el.accuracy)


def test(rate, batch_size, epochs):
    test_x, test_y = retrieve_data(mnist.test)

    with tf.Session() as session:
        session.run(init_op)
        steps = mnist.train.num_examples // batch_size

        for epoch in range(epochs):
            print(f"Testing... epoch: {epoch + 1}")
            for step in range(steps):
                vx, vy = mnist.train.next_batch(batch_size)
                session.run([train_op, loss], feed_dict={x: vx, y: to_categorical(vy), learning_rate: rate})

        acc = session.run(accuracy, feed_dict={x: test_x, y: test_y})
    return acc


search_parameters([0.05, 0.01, 0.005], [10, 50, 100], 200)
best_params = choose_best_parameters()
print(f"Best parameters: \n Learning rate: {best_params.learning_rate}, batch size: {best_params.batch_size},"
      f" number of epochs: {best_params.epochs}, accuracy: {best_params.accuracy}")
acc = test(best_params.learning_rate, best_params.batch_size, best_params.epochs)
print(f"Accuracy on test sample: {acc}")
