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
tf.set_random_seed(1)


from data_reader import read_file, get_data


FLAGS = None


def get_kde_models(kde_model_file):
    models = pickle.load(open(kde_model_file, 'rb'))
    return models


def get_batch(models, data, size, input_count):
    pm = models['plus_model']
    mm = models['minus_model']

    idx = np.random.randint(data.shape[0], size=size)
    data_sample = data[idx, :]

    errors = np.random.uniform(low=-0.01, high=0.01, size=(size, input_count))

    data_x = data_sample + errors

    pl = pm.score_samples(data_x)
    mn = mm.score_samples(data_x)
    
    return data_x, np.transpose(np.vstack([pl, mn]))
    

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


def print_model(FLAGS, model_name):
    inference_graph = tf.Graph()
    with tf.Session(graph=inference_graph) as sess:
        loader = tf.train.import_meta_graph(os.path.join(FLAGS.model_dir, model_name+".meta"))
        loader.restore(sess, os.path.join(FLAGS.model_dir, model_name))

        # x = tf.placeholder(tf.float32, [None, input_count])
        # y_ = tf.placeholder(tf.int32, [None, out_count])

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


        bestW1, bestW2, bestW3, bestB1, bestB2, bestB3 = sess.run([_W1, _W2, _W3, _b1, _b2, _b3])

    os.makedirs(FLAGS.model_dir, exist_ok=True)
    np.savetxt(os.path.join(FLAGS.model_dir, FLAGS.result+'w1.txt'), bestW1, delimiter=', ', newline='],\n[', header='[\n', footer='', comments='')
    np.savetxt(os.path.join(FLAGS.model_dir, FLAGS.result+'b1.txt'), bestB1, delimiter=', ', newline=',\n', header='[\n', footer=']', comments='')

    np.savetxt(os.path.join(FLAGS.model_dir, FLAGS.result+'w2.txt'), bestW2, delimiter=', ', newline='],\n[', header='[\n[', footer=']', comments='')
    np.savetxt(os.path.join(FLAGS.model_dir, FLAGS.result+'b2.txt'), bestB2, delimiter=', ', newline=',\n', header='[\n', footer=']', comments='')

    np.savetxt(os.path.join(FLAGS.model_dir, FLAGS.result+'w3.txt'), bestW3, delimiter=', ', newline='],\n[', header='[\n[', footer=']', comments='')
    np.savetxt(os.path.join(FLAGS.model_dir, FLAGS.result+'b3.txt'), bestB3, delimiter=', ', newline=',\n', header='[\n', footer=']', comments='')



def main(_):
    kde_models = get_kde_models(FLAGS.kde_model)

    data = get_data(FLAGS.data)
    names = [  # "current_slice",
        "blue", "blue_1",
        "green", "green_1",
        "red", "red_1",
        "nir", "nir_1",
        "swir1", "swir1_1",
        "swir2", "swir2_1"
    ]
    data = np.array(data[names])

    out_count = 2
    input_count = 12

    x = tf.placeholder(tf.float32, [None, input_count], name='input')

    with tf.name_scope('weights'):
        W1 = tf.Variable(tf.truncated_normal([input_count, FLAGS.layer1], stddev=0.5), name='w1')
        W2 = tf.Variable(tf.truncated_normal([FLAGS.layer1, FLAGS.layer2], stddev=0.5), name='w2')
        W3 = tf.Variable(tf.truncated_normal([FLAGS.layer2, out_count], stddev=0.5), name='w3')

    with tf.name_scope('biases'):
        b1 = tf.Variable(tf.zeros([FLAGS.layer1]), name='b1')
        b2 = tf.Variable(tf.zeros([FLAGS.layer2]), name='b2')
        b3 = tf.Variable(tf.zeros([out_count]), name='b3')

    y = model(x, W1, W2, W3, b1, b2, b3)



    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, out_count], name='target')

    loss1 = tf.reduce_mean(
        tf.losses.absolute_difference(labels=y_, predictions=y), name='loss1')
    tf.summary.scalar('abs diff', loss1)

    # reg_w = 0.00000001
    loss = tf.reduce_mean(
        tf.losses.mean_squared_error(labels=y_, predictions=y)
            # + reg_w*(tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2)+tf.nn.l2_loss(W3)),
        , name='loss'
    )

    tf.summary.scalar('Regularized loss', loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    optimizer = optimizer.minimize(loss)

    saver = tf.train.Saver(max_to_keep=1)

    sess = tf.InteractiveSession()

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)

    tf.global_variables_initializer().run()
    # Train
    for epoch in range(FLAGS.max_epoch):
        # получение выборки данных - очень дорогая операция Поэтому будем использовать данные многократно
        # и, чтобы меньше переобучаться, сразу много
        batch_xs, batch_ys = get_batch(kde_models, data, 384*50, input_count)
        for i in range(5000):
            _ = sess.run(
                [optimizer], feed_dict={x: batch_xs, y_: batch_ys}
            )

        # Test trained model
        # if epoch % 100 == 99:
        summary, train_loss, train_loss1 = sess.run(
            [merged, loss1, loss], feed_dict={x: batch_xs, y_: batch_ys}
        )
        print('EPOCH', epoch+1, '\tloss', train_loss, '\tloss1', train_loss1)
        train_writer.add_summary(summary, epoch)

        saver.save(sess, os.path.join(FLAGS.model_dir, "model"))

    print_model(FLAGS, 'model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kde_model', type=str, default='model.pkl',
                        help='FileName of stored model')
    parser.add_argument('--data', type=str, default='../../data/changes/expectedVSmedian/change_sample_*_seed*.csv',
                        help='Pattern for describe input data files')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory for store trained models files')
    parser.add_argument('--summaries_dir', type=str, default='summaries',
                        help='Directory for store summary log files')

    parser.add_argument('--result', type=str, default='results_',
                        help='Prefix for nnet weights')
    parser.add_argument('--layer1', type=int, default='15',
                        help='Neuron count of the first layer')
    parser.add_argument('--layer2', type=int, default='15',
                        help='Neuron count of the first layer')
    parser.add_argument('--max_epoch', type=int, default='1000',
                        help='Neuron count of the first layer')



    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


