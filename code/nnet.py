# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import glob

import numpy as np
import pandas as pd

import tensorflow as tf

tf.set_random_seed(1)

FLAGS = None

# Все завязано на конкретный формат файла. При необходимости - менять функцию
def read_file(filename, no_change=28):
    #     system:index,
    #     blue,blue_1,blue_2,
    #     current_slice,found_slice,
    #     green,green_1,green_2,
    #     id,
    #     nir,nir_1,nir_2,red,red_1,red_2,swir1,swir1_1,swir1_2,swir2,swir2_1,swir2_2,
    #     .geo
    names = ("id",
              "found_slice", "current_slice",
              "blue", "blue_1", "blue_2",
              "green", "green_1", "green_2", 
              "nir", "nir_1", "nir_2", 
              "red", "red_1", "red_2", 
              "swir1", "swir1_1", "swir1_2", 
              "swir2", "swir2_1", "swir2_2")
    formats = ('S64', 
                'i4', 'i4', 
                'f4','f4','f4',  
                'f4','f4','f4', 
                'f4','f4','f4',   
                'f4','f4','f4',   
                'f4','f4','f4',  
                'f4','f4','f4',)
    data = pd.read_csv(
        filename, 
        delimiter=',',
        # dtype={
        #     'names': names,
        #     'formats': formats
        # },
        #skiprows=1,
        usecols=[9, 5, 4, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    )
    
    data['blue_2'] = data['blue_2'] / 10000.0
    data['green_2'] = data['green_2'] / 10000.0
    data['red_2'] = data['red_2'] / 10000.0
    data['nir_2'] = data['nir_2'] / 10000.0
    data['swir1_2'] = data['swir1_2'] / 10000.0
    data['swir2_2'] = data['swir2_2'] / 10000.0
    data['old'] = data['current_slice'] - data['found_slice']
    data.loc[data['found_slice'] < 0, 'old'] = no_change
    data.loc[data['old'] < 0, 'old'] = no_change
    
    return data[list(names)+['old']]

def get_data(pattern):
    fnames = glob.glob(pattern)
    df_lst = [read_file(f) for f in fnames]
    
    return pd.concat(df_lst)

def split_data(data, train_val_test=(0.66, 0.17, 0.17), seed=0):
    np.random.seed(seed)
    ids = set(data['id'])
    count = len(ids)
    
    train_count = int(count*train_val_test[0])
    val_count = int(count*train_val_test[1])
    # test_count = int(count*train_val_test[2])
    
    train = np.random.choice(list(ids), train_count, False)
    
    test_val_ids = ids.difference(set(train))
    val = np.random.choice(list(test_val_ids), val_count, False)
    
    test = np.array(list(test_val_ids.difference(set(val))))
      
    train = data[data['id'].isin(train)]
    val = data[data['id'].isin(val)]
    test = data[data['id'].isin(test)]
    
    names = [ # "current_slice",
              "blue", "blue_1", "blue_2",
              "green", "green_1", "green_2", 
              "nir", "nir_1", "nir_2", 
              "red", "red_1", "red_2", 
              "swir1", "swir1_1", "swir1_2", 
              "swir2", "swir2_1", "swir2_2", "old"]

    names = [ # "current_slice",
              "blue_1", "blue_2",
              "green_1", "green_2",
              "nir_1", "nir_2",
              "red_1", "red_2",
              "swir1_1", "swir1_2",
              "swir2_1", "swir2_2", "old"]

    train = np.array(train[names])
    val = np.array(val[names])
    test = np.array(test[names])
    
    # np.random.shuffle(train)
    # np.random.shuffle(val)
    # np.random.shuffle(test)
    
    return train, val, test


def gv_process(g, v):
    # g = tf.Print(g, [g], 'G(before): ')
    g2 = tf.zeros_like(g, dtype=tf.float32)
    v2 = tf.zeros_like(v, dtype=tf.float32)
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

    l1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    l2 = tf.nn.relu(tf.matmul(l1, W2) + b2)

    l3_logits = tf.matmul(l2, W3) + b3

    return l3_logits


def main(_):
    # Import data
    data = get_data(FLAGS.data)
    train, val, test = split_data(data)
    print(val.shape)

    out_count = 1
    input_count = train.shape[1] - out_count

    global_step = tf.Variable(0, trainable=False)

    x = tf.placeholder(tf.float32, [None, input_count])

    W1 = tf.Variable(tf.truncated_normal([input_count, FLAGS.layer1], stddev=0.5))
    W2 = tf.Variable(tf.truncated_normal([FLAGS.layer1, FLAGS.layer2], stddev=0.5))
    W3 = tf.Variable(tf.truncated_normal([FLAGS.layer2, out_count], stddev=0.5))

    b1 = tf.Variable(tf.zeros([FLAGS.layer1]))
    b2 = tf.Variable(tf.zeros([FLAGS.layer2]))
    b3 = tf.Variable(tf.zeros([out_count]))

    logits = model(x, W1, W2, W3, b1, b2, b3)
    y = tf.exp(logits)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, out_count])

    loss1 = tf.reduce_mean(
        tf.losses.absolute_difference (labels=y_, predictions=y))
    loss = tf.reduce_mean(
        tf.nn.log_poisson_loss(targets=y_, log_input=logits))

    optimizer = training(loss, learning_rate=0.01, momentum=0.02)

    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()
    # Train
    best_loss = 9999999999999;
    for epoch in range(FLAGS.max_epoch):
        batch_xs, batch_ys = get_batch(train, 2**13, input_count)
        _, train_loss, train_loss1 = sess.run(
            [optimizer, loss, loss1], feed_dict={x: batch_xs, y_: batch_ys}
        )

        # Test trained model
        if epoch % 100 == 99:
            val_loss = sess.run(loss, feed_dict={x: val[:, :input_count], y_: val[:, input_count:]})
            if val_loss < best_loss:
                best_loss = val_loss
                bestW1, bestW2, bestW3, bestB1, bestB2, bestB3 = sess.run([W1, W2, W3, b1, b2, b3])
            # print('EPOCH', epoch+1, 'Loss: \tval', val_loss, '\ttrain', train_loss)
            print('EPOCH', epoch+1, 'Loss: \tval', sess.run(loss1, feed_dict={x: val[:, :input_count], y_: val[:, input_count:]}), '\ttrain', train_loss1)


    print('test ACC:\t', sess.run(loss, feed_dict={x: test[:, :input_count], y_: test[:, input_count:]}))
    print('test ACC1:\t', sess.run(loss1, feed_dict={x: test[:, :input_count], y_: test[:, input_count:]}))

    print('W1:\n', bestW1)
    print('b1:\n', bestB1,)
    np.savetxt(FLAGS.result+'w1.txt', bestW1, delimiter=', ')
    np.savetxt(FLAGS.result+'b1.txt', bestB1, delimiter=', ')
    
    print('W2:\n', bestW2)
    print('b2:\n', bestB2)
    np.savetxt(FLAGS.result+'w2.txt', bestW2, delimiter=', ')
    np.savetxt(FLAGS.result+'b2.txt', bestB2, delimiter=', ')
    
    print('W3:\n', bestW3)
    print('b3:\n', bestB3)
    np.savetxt(FLAGS.result+'w3.txt', bestW3, delimiter=', ')
    np.savetxt(FLAGS.result+'b3.txt', bestB3, delimiter=', ')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/home/klsvd/laboro/NextGIS/AML/DDD-alarm/EE-timeseries/change_sample_prev*_seed0.csv',
                        help='Directory for storing input data')
    parser.add_argument('--result', type=str, default='results_',
                        help='Prefix for nnet weights')
    parser.add_argument('--layer1', type=int, default='15',
                        help='Neuron count of the first layer')
    parser.add_argument('--layer2', type=int, default='15',
                        help='Neuron count of the first layer')
    # parser.add_argument('--layer3', type=int, default='15',
                        # help='Neuron count of the first layer')
    parser.add_argument('--max_epoch', type=int, default='1000',
                        help='Neuron count of the first layer')


    # tmp = get_data('/home/klsvd/laboro/NextGIS/AML/DDD-alarm/EE-timeseries/change_sample_prev*_seed0.csv')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


