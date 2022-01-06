#!/usr/bin/env python
# coding: utf-8

# This file is multi-view and wide&deep model

# Author: MA Jun
# Date: 2019.12.22
# Date: 2021.12.31

from __future__ import absolute_import, division, print_function
import argparse
import time
import os
import tensorflow as tf
from numpyIO import NumpyIO
print(tf.version)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable GPU info
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print("Is there a GPU available: ")
# assert print(device_lib.list_local_devices()), "No GPUs found."
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.67)


# Read the numpy dataset
npIO = NumpyIO("./train_X.pkl", "./test_X.pkl", "./train_y.dat", "./test_y.dat")
dataset = npIO.load()
train_dataset, test_dataset = npIO.getDataset()
num_classes = npIO.num_classes

# table: 0-86, 87-149, 150-164, 165-229, 230-261
Viewer = {
    0: range(0, 87),
    1: range(87, 150),
    2: range(150, 165),
    3: range(165, 230),
    4: range(230, 261)
}


# Define the Flags, Activations, ModeKeys and Optimizers
FLAGS = tf.flags.FLAGS
FLAGS.__dict__.update()
try:
    tf.flags.DEFINE_integer("batch_size", 32, "batch_size")
    tf.flags.DEFINE_string("hidden_activation", "tanh", "hidden activation")
    tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
    tf.flags.DEFINE_string('modelDir', './xx_models', 'model dir')
    tf.flags.DEFINE_string('checkpointDir', 'xx', 'oss info')
    tf.flags.DEFINE_string('outputs', 'predict_result', "predict saved name")
    tf.flags.DEFINE_integer('num_steps', 256, 'training steps')
    tf.flags.DEFINE_string('optimizer', 'sgd', 'optimizer methods')
    tf.flags.DEFINE_string('activation', 'bound', 'activation methods')
    tf.flags.DEFINE_bool('dropout', False, 'dropout')
    tf.flags.DEFINE_bool('norm', False, 'l2normalization')
    tf.flags.DEFINE_bool('attention', False, 'attention')
    tf.flags.DEFINE_string("predict_type", "output", "predict type")
    tf.flags.DEFINE_string('predict_checkpoint', '', 'predict_checkpoint')
    tf.flags.DEFINE_string('emb_combiner', 'mean', 'emb_combiner')
    tf.flags.DEFINE_integer("word_vocabulary_size", 50000, "word vocabulary size")
    tf.flags.DEFINE_integer("emb_dim", 64, 'embedding dim')
    tf.flags.DEFINE_integer("align_size", 32, " align and attention size")
    tf.flags.DEFINE_integer("item_net_layer", 5, "net layer")
    tf.flags.DEFINE_float('reg_weight', 0.01, 'reg weight')
    tf.flags.DEFINE_string('phase', 'train', "train or predict or delete")
    tf.flags.DEFINE_string('f', '', 'kernel')
    tf.flags.DEFINE_integer('num_classes', num_classes, 'number of classes')
    tf.flags.DEFINE_string('summaries_dir', './tensorboard', 'tensorboard dir')
except argparse.ArgumentError:
    pass

Activation = {
    'relu': tf.nn.relu,
    'sigmoid': tf.sigmoid,
    'tanh': tf.tanh,
    'identity': tf.identity,
    'none': None,
    'bound': lambda x: x - tf.nn.relu(x - 1.0) + tf.nn.relu(-x - 1.0)
}

ModeKeys = {
    'train': tf.estimator.ModeKeys.TRAIN,
    'eval': tf.estimator.ModeKeys.EVAL,
    'predict': tf.estimator.ModeKeys.PREDICT
}

Optimizer = {
    'adam': tf.train.AdamOptimizer,
    'sgd': tf.train.GradientDescentOptimizer,
    'adagrad': tf.train.AdagradOptimizer
}


def input_fn(is_train, np_data):
    """Fetch input batches
    
    Args:
        is_train: bool
        np_data: numpy
    
    Returns:
        feature tensors and label tensors
        
    Raises:
        IOError: an error occurred accessing the bigtable
    """

    # extraction
    try:
        # from_tensor_slices accepts tuple, dict, numpy
        num_col = np_data.shape[1]
        name_ss = [str(i) for i in range(num_col-1)]
        data_set = tf.data.Dataset.from_tensor_slices((dict(zip(name_ss, np_data[:-1])), np_data[-1]))
    except Exception as e:
        tf.logging.error('input error!', e)
        table_path = ['./train_X.pkl', './test_X.pkl', './train_y.dat', './test_y.dat']
        data_set = tf.data.Dataset.from_tensor_slices(table_path)
    tf.logging.info("input done!")

    # transformation
    if is_train:
        pair_data_set = data_set.shuffle(10000).repeat().batch(FLAGS.batch_size, drop_remainder=False)
    else:
        pair_data_set = data_set.batch(FLAGS.batch_size, drop_remainder=False)

    # loading
    pair_data = pair_data_set.make_one_shot_iterator().get_next()
    for key, item in pair_data[0].items():
        pair_data[0][key] = tf.to_float(item)  # float32
    features = pair_data[0]  # tuple
    labels = tf.to_float(pair_data[1])

    print("features %s" % features.keys())  # dict()
    print("labels %s" % labels)

    return features, labels


def __train_input_fn():
    return input_fn(is_train=True, np_data=train_dataset)


def __test_input_fn():
    return input_fn(is_train=False, np_data=test_dataset)


# Store and index categorical and numerical columns
CATEGORICAL_COLUMNS = [dict() for _ in range(5)]
NUMERICAL_COLUMNS = [dict() for _ in range(5)]


def __is_category(np_feature):
    """return if it is a categorical tf column
    the numpy dataset's dtype is heterogeneous data with mix of int64, float64 or object
    """
    cc_set = set(np_feature)
    cc_values = list(cc_set)
    cc_max = max(cc_values)
    cc_len = len(cc_values)
    if cc_max + 1 == cc_len:
        return True
    else:
        return False


for j in range(5):  # five views
    for cc in Viewer[j]:
        if __is_category(dataset[:, cc]):  # numpy
            cc_size = max(dataset[:, cc]) + 1
            CATEGORICAL_COLUMNS[j][str(cc)] = int(cc_size)  # key is str cc and value is int cc_size
        else:
            NUMERICAL_COLUMNS[j][str(cc)] = -1


params = {'learning_rate': FLAGS.learning_rate, 'optimizer': Optimizer[FLAGS.optimizer],
          'activation': Activation[FLAGS.activation],
          'regularizer': tf.contrib.layers.l2_regularizer(FLAGS.reg_weight)}


def __align(inputs, dim1, dim2):
    """
    align them into same dimension
    :param inputs: list
    :param dim1:
    :param dim2:
    :return: list
    """
    inputs = inputs
    for t1 in range(len(inputs)):
        v1 = tf.Variable(tf.random_normal([dim1, inputs[t1].shape[1].value], stddev=0.1), name='dim1')
        inputs[t1] = tf.matmul(v1, inputs[t1], name='align1')
    for t2 in range(len(inputs)):  # inputs[t2].shape = [:, dim1, inputs[t2].shape[2]]
        v2 = tf.Variable(tf.random_normal([inputs[t2].shape[2].value, dim2], stddev=0.1), name='dim2')
        inputs[t2] = tf.matmul(inputs[t2], v2, name='align2')
    return inputs  # shape = [:, dim1, dim2]


def __attention(inputs, attention_size):
    """
    the basic attention definition
    :param inputs: list
    :param attention_size:
    :return: tensor[:, word_size*attention_size, attention_size]
    """
    inputs = [xx for xx in inputs if xx is not None]
    inputs = __align(inputs, attention_size, attention_size)
    key = tf.Variable(tf.random_normal([attention_size, attention_size], stddev=0.1), name='w')
    q = tf.concat(inputs, axis=1)
    print("q shape is ", q.shape)
    print("On GPU: ")
    with tf.device("GPU:0"):  # Force execution on GPU
        # attention stage one
        v = tf.tanh(tf.tensordot(q, key, axes=1, name='v'))
        print("v shape is ", v.shape)
        # attention stage two
        alpha = tf.nn.softmax(v, name='a')
        print("alpha shape is ", alpha.shape)
        return alpha


def model_fn(features, labels, mode):
    """
    build nn-based multi-view architecture
    :param features: dtype is dict(), so index column with features[str(key)]
    :param features: meanwhile, embedding_lookup's ids is tf.int32 or tf.int64
    :param labels:
    :param mode:
    :return:
    """

    net = [dict() for _ in range(5)]  # embedding list hat e
    with tf.name_scope('embedding'):

        cc_embed_ding = [dict() for _ in range(5)]
        tmp_net = [list() for _ in range(5)]

        for k in range(5):

            # build cate net
            if CATEGORICAL_COLUMNS[k]:

                # ids embedding
                word_size = sum(CATEGORICAL_COLUMNS[k].values())
                cc_embed_var = tf.Variable(tf.random_uniform([word_size, FLAGS.emb_dim], -0.1, 0.1), name='cate_embedding_var')

                for key, value in CATEGORICAL_COLUMNS[k].items():

                    # word embedding
                    cc_embed_ding[k][key] = tf.nn.embedding_lookup(cc_embed_var, tf.cast(features[str(key)], dtype=tf.int32), name='embedding_feature')

                # concat cate net
                cate_keys = sorted(CATEGORICAL_COLUMNS[k].keys())
                feature_embed_net = tf.concat([tf.expand_dims(cc_embed_ding[k][key], 1) for key in cate_keys], axis=1)
                print('concat embed feature columns shape is', feature_embed_net.shape)
                tmp_net[k].append(feature_embed_net)

            # build num net
            if NUMERICAL_COLUMNS[k]:

                # concat num net
                num_keys = sorted(NUMERICAL_COLUMNS[k].keys())
                feature_num_net = tf.expand_dims(tf.concat([tf.to_float(tf.reshape(features[str(key)], [-1, 1])) for key in num_keys], axis=1), axis=2)
                print('concat num feature columns shape is', feature_num_net.shape)
                tmp_net[k].append(feature_num_net)

            # merge
            net[k] = tf.concat(tmp_net[k], axis=1) if len(tmp_net[k]) >= 2 else tmp_net[k][0]

    with tf.name_scope('hidden'):

        def wide_net_func(inputs, reuse):
            layer = tf.reduce_sum(inputs, axis=1)
            with tf.variable_scope('hidden/prob_net', reuse=reuse) as vs1:
                print("vs1 is ", vs1)
                layer = tf.contrib.layers.fully_connected(
                    layer,
                    FLAGS.num_classes,
                    scope='wide_net_l%s' % 1,
                    weights_regularizer=params['regularizer'],
                    biases_regularizer=params['regularizer'],
                    activation_fn=params['activation'])
            return layer

        def deep_net_func(inputs, reuse):
            layer = tf.reduce_sum(inputs, axis=1)
            with tf.variable_scope('hidden/item_net', reuse=reuse) as vs2:
                print("vs2 is ", vs2)
                for i in range(1, FLAGS.item_net_layer + 1):
                    if i == 1:
                        layer = params['activation'](layer)
                    else:
                        layer = tf.contrib.layers.fully_connected(
                            layer,
                            FLAGS.emb_dim if i < FLAGS.item_net_layer else FLAGS.num_classes,
                            scope='item_net_l%s' % i,
                            weights_regularizer=params['regularizer'],
                            biases_regularizer=params['regularizer'],
                            activation_fn=params['activation'])
            return layer

    with tf.name_scope('logits'):

        # attention
        d_net = __attention([net[t] for t in range(5)], FLAGS.align_size)

        # dropout
        if FLAGS.dropout:
            d_net = tf.layers.dropout(d_net, rate=0.1, training=False, name='Dropout')
            print('dropout net shape is ', d_net.shape)

        # normalization
        if FLAGS.norm:
            net_norm = tf.sqrt(tf.reduce_sum(tf.square(d_net), axis=1, keep_dims=True) + 1e-8)
            d_net = tf.truediv(d_net, net_norm, name='truediv')
            print('normalized net shape is ', d_net.shape)

        # wide
        net = __align(net, FLAGS.align_size, FLAGS.align_size)
        w_net = tf.concat(net, axis=1)
        print("w_net shape is ", w_net.shape)
        wide_logits = wide_net_func(w_net, tf.AUTO_REUSE)

        # deep
        print("d_net shape is ", d_net.shape)
        deep_logits = deep_net_func(d_net, tf.AUTO_REUSE)

        logits = wide_logits + deep_logits
        print('logits shape is ', logits.shape)
        tf.print(logits, [logits], "tf.print: logits ")

        pre_classes = tf.argmax(logits, axis=1)
        pre_prob = tf.nn.softmax(logits, name='classification_predict_prob')
        print("predicted probability is ", pre_prob)

    with tf.name_scope('loss'):

        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.cast(labels, dtype=tf.int32)))

        loss_sy = tf.summary.histogram('cross_entropy', loss_op)

        optimizer = params['optimizer'](learning_rate=FLAGS.learning_rate)
        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

        acc_op = tf.metrics.accuracy(labels=labels, predictions=pre_classes)

        acc_sy = tf.summary.histogram('accuracy', acc_op)

        # merge all summaries and write them out to disk
        summaries = tf.summary.merge([loss_sy, acc_sy])
        print(summaries)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=pre_classes)

        estimator_specs = tf.estimator.EstimatorSpec(
            mode=mode, predictions=pre_classes, loss=loss_op, train_op=train_op, eval_metric_ops={'accuracy': acc_op})

        return estimator_specs


"""Configuration before training"""
model_dir = os.path.join(FLAGS.modelDir)
config = tf.estimator.RunConfig(
    model_dir=model_dir
)

MvWDL = tf.estimator.Estimator(model_fn=model_fn)

"""Train"""
MvWDL.train(input_fn=__train_input_fn, steps=FLAGS.num_steps)
print("Training costs time: {}", time.time())

"""Evaluate"""
result = MvWDL.evaluate(input_fn=__test_input_fn)
print("testing accuracy: ", result['accuracy'])
print("mean loss per mini-batch: ", result['loss'])

"""Save estimator model"""
serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(input_fn)

path = MvWDL.export_saved_model(export_dir_base='./export_saved_model',
                                serving_input_receiver_fn=serving_input_receiver_fn,
                                as_text=False)
