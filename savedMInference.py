#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import argparse
# GENERAL DEFINES
INFERENCE_GRAPH = 'graph.pbtxt'

# Define the Flags, Activations, ModeKeys and Optimizers.
FLAGS = tf.flags.FLAGS
FLAGS.__dict__.update()
try:
    tf.flags.DEFINE_string('modelDir', './xx_models', 'model dir')
    tf.flags.DEFINE_string('checkpointDir', 'xx', 'oss info')
except argparse.ArgumentError:
    pass


"""Load graph"""
f = open(FLAGS.modelDir + '/' + INFERENCE_GRAPH, 'rb')
gd = tf.GraphDef.FromString(f.read())
tf.import_graph_def(gd, name='')

"""Run the inference graph and extract middle layer"""
with tf.Session() as sess:
    values = tf.get_default_graph().get_operation_by_name("hidden/item_net:0").outputs[0]
    middle_p = sess.run(values)
print("middle results of prob P is: {}".format(middle_p))
