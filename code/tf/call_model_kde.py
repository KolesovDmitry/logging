# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import pickle

import numpy as np
import tensorflow as tf

from data_reader import read_file, get_data
from net4kde import get_kde_models

FLAGS = None


def eval_kde(models, data):
    pm = models['plus_model']
    mm = models['minus_model']

    pl = pm.score_samples(data)
    mn = mm.score_samples(data)

    return np.transpose(np.vstack([pl, mn]))


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

        _loss = inference_graph.get_tensor_by_name('loss:0')
        _loss1 = inference_graph.get_tensor_by_name('loss1:0')
        _x = inference_graph.get_tensor_by_name('input:0')
        _y = inference_graph.get_tensor_by_name('target:0')

        _W1 = inference_graph.get_tensor_by_name('weights/w1:0')
        _W2 = inference_graph.get_tensor_by_name('weights/w2:0')
        _W3 = inference_graph.get_tensor_by_name('weights/w3:0')

        _b1 = inference_graph.get_tensor_by_name('biases/b1:0')
        _b2 = inference_graph.get_tensor_by_name('biases/b2:0')
        _b3 = inference_graph.get_tensor_by_name('biases/b3:0')

        result = model(_x, _W1, _W2, _W3, _b1, _b2, _b3)

        res = sess.run(result, feed_dict={_x: x_test})

        return res



def main(_):
    input_count = 12
    kde_models = get_kde_models(FLAGS.kde_model)


    data = get_data(FLAGS.data)
    plus = data.loc[(data['change'] == 1)]
    minus = data.loc[(data['change'] == 0)]

    names = [  # "current_slice",
        "blue", "blue_1",
        "green", "green_1",
        "red", "red_1",
        "nir", "nir_1",
        "swir1", "swir1_1",
        "swir2", "swir2_1"
    ]

    plus = np.array(plus[names])
    minus = np.array(minus[names])

    # plus_sample = plus[:10, :]
    # minus_sample = minus[:10, :]

    # expected = eval_kde(kde_models, plus_sample)
    # recived = call_model(FLAGS, 'model', plus_sample)

    # print(np.hstack([expected, recived]))

    # expected = eval_kde(kde_models, minus_sample)
    # recived = call_model(FLAGS, 'model', minus_sample)
    # print(np.hstack([expected, recived]))

    density_plus = call_model(FLAGS, 'model', plus)
    print(density_plus)
    density_minus = call_model(FLAGS, 'model', minus)
    print(density_minus)

    true_plus = density_plus[:, 0] > density_plus[:, 1]
    true_plus_prop = 1.0 * sum(true_plus.astype(np.int)) / len(true_plus)

    true_minus = density_minus[:, 1] > density_minus[:, 0]
    true_minus_prop = 1.0 * sum(true_minus.astype(np.int)) / len(true_minus)

    print('Plus prob', true_plus_prop)
    print('Minus prob', true_minus_prop)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kde_model', type=str, default='model.pkl',
                        help='FileName of stored model')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory for store trained models files')
    parser.add_argument('--data', type=str, default='../../data/changes/expectedVSmedian/change_sample_*_seed*.csv',
                        help='Pattern for describe input data files')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


