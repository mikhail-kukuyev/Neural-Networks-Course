#!/usr/bin/env python3
from __future__ import print_function
import argparse

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def hidden_cnn_layers(input_layer, mode):
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.initializers.he_normal())

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    norm1 = tf.layers.batch_normalization(inputs=pool1)
    dropout1 = tf.layers.dropout(inputs=norm1, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

    conv2 = tf.layers.conv2d(
        inputs=dropout1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.initializers.he_normal())

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    dropout2 = tf.layers.dropout(inputs=pool2, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

    flat = tf.reshape(dropout2, [-1, 7 * 7 * 64])

    dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
    dropout3 = tf.layers.dropout(inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    return dropout3


def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    hidden = hidden_cnn_layers(input_layer, mode)

    logits = tf.layers.dense(inputs=hidden, units=10)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Loss for TRAIN and EVAL modes
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Add accuracy evaluation for both TRAIN and EVAL modes
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"], name='acc_op')
    tf.summary.scalar('accuracy', accuracy[1])

    # Configure the Training Op
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    metrics = {"accuracy": accuracy}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)


def train(data_dir, model_dir, batch_size, steps):
    mnist = input_data.read_data_sets(data_dir, validation_size=0)
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int)

    classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=model_dir)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    classifier.train(input_fn=train_input_fn, steps=steps)


def validate(data_dir, model_dir, batch_size, steps, validation_step):
    mnist = input_data.read_data_sets(data_dir, validation_size=5000)
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int)
    eval_data = mnist.validation.images
    eval_labels = np.asarray(mnist.validation.labels, dtype=np.int)

    classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=model_dir)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    for _ in range(steps // validation_step):
        classifier.train(input_fn=train_input_fn, steps=validation_step)
        eval_results = classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)


def test(data_dir, model_dir):
    mnist = input_data.read_data_sets(data_dir)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int)

    classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=model_dir)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    return classifier.evaluate(input_fn=eval_input_fn)


def parse_args():
    parser = argparse.ArgumentParser(prog='trains and evaluates CNN on fashion-mnist dataset')
    parser.add_argument('mode', choices=['train', 'test', 'validate'],
                        help="program mode, use 'train' for training model, 'validate' for validation"
                             " and 'test' for evaluation on the test set")
    parser.add_argument('model_dir', help='path to folder where trained model is stored or should be saved,'
                                          ' depending on the specified mode')
    parser.add_argument('data_dir', help='path to folder with train, test data or both of them, depending on the specified mode')
    return parser.parse_args()


def main(argv):
    args = parse_args()

    if args.mode == 'train':
        train(args.data_dir, args.model_dir, batch_size=256, steps=6000)
        print(f"Model was successfully trained and saved in '{args.model_dir}' directory.")

    elif args.mode == 'validate':
        validate(args.data_dir, args.model_dir, batch_size=256, steps=10000, validation_step=50)
        print(f"Model was successfully trained, validated and saved in '{args.model_dir}' directory.")

    elif args.mode == 'test':
        test_results = test(args.data_dir, args.model_dir)
        print(f"Test accuracy: {test_results['accuracy']}")
        print(f"Test loss: {test_results['loss']}")
        # 92.9% accuracy was reached


if __name__ == "__main__":
    parse_args()
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.app.run()
