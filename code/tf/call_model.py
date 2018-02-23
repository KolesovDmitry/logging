# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

import numpy as np

import tensorflow as tf


FLAGS = None

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

    var_count = 13
    out_count = 1
    input_count = var_count - out_count


    # x = tf.placeholder(tf.float32, [None, input_count], name='input')

    x = np.array([0.5] * input_count) - np.random.random((1, input_count))
    print('Input:', x)

    call_model(FLAGS, 'model', x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory for store trained models files')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


