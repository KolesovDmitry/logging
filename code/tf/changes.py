# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os


import numpy as np

import tensorflow as tf

tf.set_random_seed(1)


from data_reader import read_file, get_data, split_data

FLAGS = None



def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def gv_process(g, v):
    tf.summary.histogram('grad', g)
    # g = tf.Print(g, [g], 'G(before): ')
    g2 = tf.zeros_like(g, dtype=tf.float32)
    v2 = tf.zeros_like(v, dtype=tf.float32)
    # g2 = g
    g2 = tf.clip_by_value(g, -1.0, 1.0)
    v2 = v
    # g2 = tf.Print(g2, [g2], 'G(after): ')
    return g2, v2


# Define training operation
def training(loss, learning_rate, momentum):
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)

    # learning_rate = tf.train.inverse_time_decay(start_learning_rate, global_step, decay_steps=1, decay_rate=decay_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(start_learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    gv2 = [gv_process(gv[0], gv[1]) for gv in grads_and_vars]
    train_op = optimizer.apply_gradients(gv2)
    return train_op


def get_batch(data, size, input_count):
    batch = data[np.random.randint(data.shape[0], size=size), :]
    
    return batch[:, :input_count], batch[:, input_count:]
    

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


def print_model(FLAGS, model_name, x_test, y_test):
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



        print('test ACC:\t', sess.run(_loss, feed_dict={_x: x_test,_y: y_test}))
        print('test ACC1:\t', sess.run(_loss1, feed_dict={_x: x_test,_y: y_test}))

        bestW1, bestW2, bestW3, bestB1, bestB2, bestB3 = sess.run([_W1, _W2, _W3, _b1, _b2, _b3])

    os.makedirs(FLAGS.model_dir, exist_ok=True)
    np.savetxt(os.path.join(FLAGS.model_dir, FLAGS.result+'w1.txt'), bestW1, delimiter=', ', newline='],\n[', header='[\n', footer='', comments='')
    np.savetxt(os.path.join(FLAGS.model_dir, FLAGS.result+'b1.txt'), bestB1, delimiter=', ', newline=',\n', header='[\n', footer=']', comments='')

    np.savetxt(os.path.join(FLAGS.model_dir, FLAGS.result+'w2.txt'), bestW2, delimiter=', ', newline='],\n[', header='[\n[', footer=']', comments='')
    np.savetxt(os.path.join(FLAGS.model_dir, FLAGS.result+'b2.txt'), bestB2, delimiter=', ', newline=',\n', header='[\n', footer=']', comments='')

    np.savetxt(os.path.join(FLAGS.model_dir, FLAGS.result+'w3.txt'), bestW3, delimiter=', ', newline='],\n[', header='[\n[', footer=']', comments='')
    np.savetxt(os.path.join(FLAGS.model_dir, FLAGS.result+'b3.txt'), bestB3, delimiter=', ', newline=',\n', header='[\n', footer=']', comments='')



def main(_):
    data = get_data(FLAGS.data)
    train, val, test = split_data(data)

    out_count = 1
    input_count = train.shape[1] - out_count


    x = tf.placeholder(tf.float32, [None, input_count], name='input')

    with tf.name_scope('weights'):
        W1 = tf.Variable(tf.truncated_normal([input_count, FLAGS.layer1], stddev=0.5), name='w1')
        W2 = tf.Variable(tf.truncated_normal([FLAGS.layer1, FLAGS.layer2], stddev=0.5), name='w2')
        W3 = tf.Variable(tf.truncated_normal([FLAGS.layer2, out_count], stddev=0.5), name='w3')

    with tf.name_scope('biases'):
        b1 = tf.Variable(tf.zeros([FLAGS.layer1]), name='b1')
        b2 = tf.Variable(tf.zeros([FLAGS.layer2]), name='b2')
        b3 = tf.Variable(tf.zeros([out_count]), name='b3')

    logits = model(x, W1, W2, W3, b1, b2, b3)
    y = tf.sigmoid(logits, name='result')


    # Define loss and optimizer
    y_ = tf.placeholder(tf.int32, [None, out_count], name='target')

    loss1 = tf.reduce_mean(
        tf.losses.absolute_difference(labels=y_, predictions=y), name='loss1')
    tf.summary.scalar('abs diff', loss1)

    reg_w = 0.0001
    loss = tf.reduce_mean(
        tf.losses.sparse_softmax_cross_entropy(labels=[y_], logits=[tf.transpose([-logits, logits])]) +
            reg_w*(tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2)+tf.nn.l2_loss(W3)),
        name='loss'
    )


    tf.summary.scalar('cross entropy', loss)

    optimizer = training(loss, learning_rate=0.05, momentum=0.01)

    saver = tf.train.Saver(max_to_keep=1)

    sess = tf.InteractiveSession()

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

    tf.global_variables_initializer().run()
    # Train
    best_loss = 9999999999999;
    for epoch in range(FLAGS.max_epoch):
        batch_xs, batch_ys = get_batch(train, 2**13, input_count)
        _, summary, train_loss, train_loss1 = sess.run(
            [optimizer, merged, loss, loss1], feed_dict={x: batch_xs, y_: batch_ys.astype(np.int32)}
        )
        train_writer.add_summary(summary, epoch)

        # Test trained model
        if epoch % 100 == 99:
            val_loss = sess.run(loss, feed_dict={x: val[:, :input_count], y_: val[:, input_count:].astype(np.int32)})
            print('EPOCH', epoch+1, 'Loss: \tval', sess.run(loss1, feed_dict={x: val[:, :input_count], y_: val[:, input_count:].astype(np.int32)}), '\ttrain', train_loss1)
            test_writer.add_summary(summary, epoch)

            if val_loss < best_loss:
                best_loss = val_loss

                saver.save(sess, os.path.join(FLAGS.model_dir, "model"))

    print_model(FLAGS, 'model', test[:, :input_count], test[:, input_count:].astype(np.int32))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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


