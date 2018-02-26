# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

import numpy as np

import tensorflow as tf

from data_reader import read_file, get_data, split_data



FLAGS = None


def get_batch(data, size, input_count):
    batch = data[np.random.randint(data.shape[0], size=size), :]
    x = batch[:, :input_count]
    y = batch[:, input_count:]

    return x, y

def model(x, W1, W2, W3, b1, b2, b3):

    tf.summary.histogram('w3', W3)
    tf.summary.histogram('b3', b3)

    l1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    tf.summary.histogram('l1', l1)

    l2 = tf.nn.relu(tf.matmul(l1, W2) + b2)
    tf.summary.histogram('l2', l2)

    l3_logits = tf.matmul(l2, W3) + b3
    tf.summary.histogram('l3_logits', l3_logits)

    return l3_logits


def call_model(FLAGS, model_name, x_test):
    inference_graph = tf.Graph()
    with tf.Session(graph=inference_graph) as sess:
        loader = tf.train.import_meta_graph(os.path.join(FLAGS.model_dir, model_name+".meta"))
        loader.restore(sess, os.path.join(FLAGS.model_dir, model_name))

        _x = inference_graph.get_tensor_by_name('input:0')
        y = inference_graph.get_tensor_by_name('result:0')

        print('Result:\t', sess.run(y, feed_dict={_x: x_test}))



def main(_):
    data = get_data(FLAGS.data)
    train, val, test = split_data(data)

    out_count = 1
    input_count = train.shape[1] - out_count

    batch_xs, batch_ys = get_batch(train, 2 ** 4, input_count)
    call_model(FLAGS, 'model', batch_xs)
    print(batch_ys)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory for store trained models files')
    parser.add_argument('--data', type=str, default='../../data/changes/expectedVSmedian/change_sample_*_seed*.csv',
                        help='Pattern for describe input data files')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


