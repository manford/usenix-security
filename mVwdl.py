#!/usr/bin/env python
# coding: utf-8

# This file is multi-view and wide&deep model

# Author: MA Jun
# Date: 2019.12.22

from __future__ import absolute_import, division, print_function
import argparse
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import time
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
print(tf.version)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable GPU info
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print("Is there a GPU available: ")
# assert print(device_lib.list_local_devices()), "No GPUs found."
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.67)

"""load raw data"""
train_X = pd.read_pickle("./train_X.pkl")
test_X = pd.read_pickle("./test_X.pkl")
train_Y = np.loadtxt("./train_y.dat", dtype=int)
test_Y = np.loadtxt("./test_y.dat", dtype=int)
print("Data loading is done.")


"""total data set"""
train_D = np.hstack((train_X.values, train_Y.reshape(-1, 1)))
print(len(train_D))
test_D = np.hstack((test_X.values, test_Y.reshape(-1, 1)))
print(len(test_D))
total_D = np.vstack((train_D, test_D))
print("The total data set shape before the pre-processing is: \n", total_D.shape)


"""introduce one-hot labels"""
label_enc = LabelEncoder()
one_hot_enc = OneHotEncoder(categories='auto')
tmp_total_D_label = np.array(label_enc.fit_transform(total_D[:, -1])).reshape(-1, 1)  # transformed labels
print("The labeled labels are: \n", tmp_total_D_label)



train_row = len(train_D)
test_row = len(test_D)
num_classes = max(np.ravel(tmp_total_D_label, 'F')) + 1
num_classes = num_classes.item()  # convert numpy.int64 to python int
print("number of classes is: \n", num_classes)

total_DD = total_D[:, :-1]  # X part
total_DDD = np.hstack((total_DD, tmp_total_D_label))  # X + y
train_D, test_D = np.split(total_DDD, [train_row])


# 0-86, 87-149, 150-164, 165-229, 230-261
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
    tf.flags.DEFINE_integer("emb_dim", 32, 'embedding dim')
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

    table_path = ['./train_X.pkl', './test_X.pkl', './train_y.dat', './test_y.dat']
    num_col = np_data.shape[1]  # column number
    names = [str(i) for i in range(num_col-1)]  # column name

    # extraction
    try:
        # from_tensor_slices accepts tuple, dict, numpy
        data_set = tf.data.Dataset.from_tensor_slices((dict(zip(names, np_data[:-1])), np_data[-1]))
        # data_set = tf.data.Dataset.from_tensor_slices({'x': np_data[:,:-1], 'y': np_data[:,-1]})
    except Exception as e:
        tf.logging.error('input error!', e)
        data_set = tf.data.Dataset.from_tensor_slices(table_path)
    tf.logging.info("input numpy done!")

    # transformation
    if is_train:
        pair_data_set = data_set.shuffle(10000).repeat().batch(FLAGS.batch_size, drop_remainder=False)
    else:
        pair_data_set = data_set.batch(FLAGS.batch_size, drop_remainder=False)

    # loading
    # pair_data_set = pair_data_set.apply(tf.contrib.data.prefetch_to_device("/gpu:0"))
    pair_data = pair_data_set.make_one_shot_iterator().get_next()
    for key, item in pair_data[0].items():
        pair_data[0][key] = tf.to_float(item)  # float32
    feature_cols = pair_data[0]  # tuple
    labels = tf.to_float(pair_data[1])  # tuple
    # feature_cols = tf.to_float(pair_data[:, :num_col-1], name='float32')  # numpy, tensor
    # labels = tf.to_float(pair_data[:, num_col-1], name='float32')  # numpy, tensor

    print("feature_cols %s" % feature_cols)
    print("labels %s" % labels)

    return feature_cols, labels


train_input_fn = lambda: input_fn(is_train=True, np_data=train_D)
test_input_fn = lambda: input_fn(is_train=False, np_data=test_D)


# store and index categorical and numerical columns

CATEGORICAL_COLUMNS = [dict() for _ in range(5)]
NUMERICAL_COLUMNS = [dict() for _ in range(5)]


def __is_category(np_feature):
    """return if it is a categorical tf column"""
    cc_values = set(np_feature)
    cc_max = max(cc_values)
    cc_len = len(cc_values)
    if cc_max + 1 == cc_len:
        return True
    else:
        return False


for j in range(5):  # five views
    for cc in Viewer[j]:
        if __is_category(total_DDD[:, cc]):
            cc_size = max(total_DDD[:, cc]) + 1
            CATEGORICAL_COLUMNS[j][int(cc)] = int(cc_size)  # key is cc and value is cc_size
        else:
            NUMERICAL_COLUMNS[j][int(cc)] = -1


params = {'learning_rate': FLAGS.learning_rate, 'optimizer': Optimizer[FLAGS.optimizer],
          'activation': Activation[FLAGS.activation],
          'regularizer': tf.contrib.layers.l2_regularizer(FLAGS.reg_weight)}


def align(inputs, dim):
    """
    align them into same dimension
    :param inputs: list
    :param dim: dimension number
    :return: list
    """
    inputs = inputs
    for tt in range(len(inputs)):
        tv = tf.Variable(tf.random_normal([inputs[tt].shape[1].value, dim], stddev=0.1), name='tv')
        inputs[tt] = tf.matmul(inputs[tt], tv, name='ne')
    return inputs


def attention(inputs, attention_size):
    """the basic attention definition"""
    inputs = [xx for xx in inputs if xx is not None]
    word_size = len(inputs)
    inputs = align(inputs, attention_size)
    # the trainable parameters
    w_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1), name='w')
    b_omega = tf.Variable(tf.random_normal([word_size], stddev=0.1), name='b')

    q = tf.reshape(tf.concat([inputs], axis=1), [-1, word_size, attention_size])
    print("q shape is ", q.shape)

    print("On GPU: ")
    with tf.device("GPU:0"): # Force execution on GPU

        v = tf.tanh(tf.tensordot(q, w_omega, axes=1, name='v') + b_omega)
        print("v shape is ", v.shape)

        alpha = tf.nn.softmax(v)
        print("alpha shape is ", alpha.shape)

        av = q * tf.expand_dims(alpha, -1)
        print("av shape is ", av.shape)

        return tf.reduce_sum(av, 1)


def model_fn(features, labels, mode):
    """build nn-based multi-view architecture"""
    # num_col = features.shape[1]
    # print("num_col is ", num_col)

    # # convert tensor to numpy
    # with tf.Session():
    #     np_features = features.eval()
    # print('numpy features are ', np_features)

    print('The features are dict type ', features)
    net = [dict() for _ in range(5)]  # embedding list hat e
    with tf.name_scope('embedding'):
        for k in range(5):
            # id embedding
            tmp_cc_embed_var = {}
            tmp_cc_embed_feature = {}
            for key, value in CATEGORICAL_COLUMNS[k].items():
                tmp_cc_embed_var[key] = tf.Variable(tf.random_uniform([value, FLAGS.emb_dim], -0.1, 0.1),
                                                    name='cate_embedding_var')
                # turn 1 dim to 2 dim
                tmp_cc_embed_feature[key] = tf.nn.embedding_lookup(tmp_cc_embed_var[key],
                                                                   tf.to_int32(tf.reshape(features[str(key)], [-1, 1])),
                                                                   partition_strategy='mod',
                                                                   name='embedding_feature')
            tmp_net = []  # collector
            # build cate net
            if CATEGORICAL_COLUMNS[k]:
                cate_keys = sorted(CATEGORICAL_COLUMNS[k].keys())
                feature_embed_net = tf.concat([tmp_cc_embed_feature[key] for key in cate_keys], axis=1)
                print('concat embed feature columns shape is', feature_embed_net.shape)
                tmp_net.append(feature_embed_net)
            # build num net
            if NUMERICAL_COLUMNS[k]:
                num_keys = sorted(NUMERICAL_COLUMNS[k].keys())
                feature_num_net = tf.concat([tf.reshape(features[str(key)], [-1, 1]) for key in num_keys], axis=1)
                print('concat num feature columns shape is', feature_num_net.shape)
                tmp_net.append(feature_num_net)
            net[k] = tf.concat(tmp_net, axis=1) if len(tmp_net) == 2 else tmp_net[0]

    with tf.name_scope('hidden'):
        def deep_net_func(inputs, reuse):
            layer = inputs
            with tf.variable_scope('hidden/item_net', reuse=reuse) as vs1:
                print("vs1 is ", vs1)
                for i in range(1, FLAGS.item_net_layer + 1):
                    num_col_ed = layer.shape[1]
                    print("temp layer shape is ", num_col_ed)
                    if i == 1:
                        layer = params['activation'](layer)
                    else:
                        layer = tf.contrib.layers.fully_connected(
                            layer,
                            FLAGS.emb_dim if (i + 1) < FLAGS.item_net_layer else FLAGS.num_classes,
                            scope='item_net_l%s' % i,
                            weights_regularizer=params['regularizer'],
                            biases_regularizer=params['regularizer'],
                            activation_fn=params['activation'] if (i + 1) < FLAGS.item_net_layer else None)
            return layer

        def wide_net_func(inputs, reuse):
            layer = inputs
            with tf.variable_scope('hidden/prob_net', reuse=reuse) as vs2:
                print("vs2 is ", vs2)
                layer = tf.contrib.layers.fully_connected(
                    layer,
                    FLAGS.num_classes,
                    scope='wide_net_l%s' % 1,
                    weights_regularizer=params['regularizer'],
                    biases_regularizer=params['regularizer'],
                    activation_fn=None)
            return layer

        attention_size = 5376
        e_net = attention([net[t] for t in range(5)], attention_size)  # attention bold e
        print('attention net shape is ', e_net.shape)

        if FLAGS.dropout:
            e_net = tf.layers.dropout(e_net, rate=0.1, training=False, name='Dropout')
            print('dropout net shape is ', e_net.shape)

        if FLAGS.norm:
            net_norm = tf.sqrt(tf.reduce_sum(tf.square(e_net), axis=1, keep_dims=True) + 1e-8)
            e_net = tf.truediv(e_net, net_norm, name='truediv')
            print('normalized net shape is ', e_net.shape)

        # wide
        net = align(net, 128)
        wide_logits = wide_net_func(net[0], tf.AUTO_REUSE)
        for o in range(1, 5):
            wide_logits += wide_net_func(net[o], tf.AUTO_REUSE)
        # deep
        deep_logits = deep_net_func(e_net, tf.AUTO_REUSE)

        logits = wide_logits + deep_logits
        print('logits net shape is ', logits.shape)
        tf.print(logits, [logits], "tf.print: logits ")

        pre_classes = tf.argmax(logits, axis=1)
        pre_prob = tf.nn.softmax(logits, name='classification_predict_prob')
        print("predicted probability is ", pre_prob)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=pre_classes)

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
        # train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train')
        # train_writer.add_summary(summaries)

        estimator_specs = tf.estimator.EstimatorSpec(
            mode=mode, predictions=pre_classes, loss=loss_op, train_op=train_op, eval_metric_ops={'accuracy': acc_op})

        return estimator_specs


"""Configuration before training"""
model_dir = os.path.join(FLAGS.modelDir)
config = tf.estimator.RunConfig(
    model_dir=model_dir
)

mVwdl = tf.estimator.Estimator(model_fn=model_fn)

"""Train"""
mVwdl.train(input_fn=train_input_fn, steps=FLAGS.num_steps)
print("Training costs time: {}".format(time.time()))

"""Evaluate"""
# trt_graph = trt.create_inference_graph(
#                 input_graph_def=frozen_graph_def,
#                 outputs=output_node_name,
#                 max_batch_size=batch_size,
#                 max_workspace_size_bytes=workspace_size,
#                 precision_mode=precision)

result = mVwdl.evaluate(input_fn=test_input_fn)
print("testing accuracy: ", result['accuracy'])
print("mean loss per mini-batch: ", result['loss'])

"""Save estimator model"""
# https://github.com/line-capstone/Projects/blob/8cf0223bbd46796a011e6c46f5985e99c2ecbaf9/References/10_FTRL(From%20Tensorflow).ipynb
# https://github.com/lontaixanh97/Tensor-Flow/blob/0a10833f24b728c98208db4847bd2ecefce4274c/SaveModel-TF.dataset/SavedModel.ipynb
# https://juejin.im/post/5ba46151e51d450e4437d1ea
# receiver is called by model_fn，so serving_feature_receiver must correct when coming into model_fn
# placeholder() is not a must，dict is a must，list is not allowed
# An input_fn that expects a serialized tf.Example
# tf.estimator has its own saved_model method

names = [str(i) for i in range(261)]  # column name

serving_feature_receiver = {
    name: tf.placeholder(tf.float32, [1], name=name + "_placeholder")
    for name in names
}

serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(serving_feature_receiver)

path = mVwdl.export_saved_model(export_dir_base='./export_saved_model',
                                serving_input_receiver_fn=serving_input_fn,
                                as_text=False)
