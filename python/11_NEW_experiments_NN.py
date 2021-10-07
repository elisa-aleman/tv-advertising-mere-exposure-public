#-*- coding: utf-8 -*-

from NomuraSoken_methods import *
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy
import errno
import os
import tensorflow as tf

######## Product Based Model
#### Purchase Models
## PrimeNotIncluded
def DM_product_purchase_cm_only_PrimeNotIncluded(run_id='', product_purchase_cm_only_PrimeNotIncluded_graph=tf.Graph()):
    n_features = 7
    with graph.as_default():
        product_purchase_cm_only_PrimeNotIncluded_net = input_data(shape=[None,n_features], name='{}_input'.format(run_id))
        product_purchase_cm_only_PrimeNotIncluded_net = fully_connected(product_purchase_cm_only_PrimeNotIncluded_net, 10, activation='relu')
        product_purchase_cm_only_PrimeNotIncluded_net = fully_connected(product_purchase_cm_only_PrimeNotIncluded_net, 10, activation='relu')
        product_purchase_cm_only_PrimeNotIncluded_net = fully_connected(product_purchase_cm_only_PrimeNotIncluded_net, 10, activation='relu')
        product_purchase_cm_only_PrimeNotIncluded_net = dropout(product_purchase_cm_only_PrimeNotIncluded_net, 0.25)
        product_purchase_cm_only_PrimeNotIncluded_net = fully_connected(product_purchase_cm_only_PrimeNotIncluded_net, 2, activation='relu')
        product_purchase_cm_only_PrimeNotIncluded_net = regression(
            product_purchase_cm_only_PrimeNotIncluded_net, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        tensorboard_log = MakeLogFile('', server=True)
        product_purchase_cm_only_PrimeNotIncluded_model = tflearn.DNN(product_purchase_cm_only_PrimeNotIncluded_net, tensorboard_dir=tensorboard_log)
    return product_purchase_cm_only_PrimeNotIncluded_model, product_purchase_cm_only_PrimeNotIncluded_graph

def DM_product_purchase_demographics_PrimeNotIncluded(run_id='', product_purchase_demographics_PrimeNotIncluded_graph=tf.Graph()):
    n_features = 27
    with graph.as_default():
        product_purchase_demographics_PrimeNotIncluded_net = input_data(shape=[None,n_features], name='{}_input'.format(run_id))
        product_purchase_demographics_PrimeNotIncluded_net = fully_connected(product_purchase_demographics_PrimeNotIncluded_net, 10, activation='relu')
        product_purchase_demographics_PrimeNotIncluded_net = fully_connected(product_purchase_demographics_PrimeNotIncluded_net, 10, activation='relu')
        product_purchase_demographics_PrimeNotIncluded_net = fully_connected(product_purchase_demographics_PrimeNotIncluded_net, 10, activation='relu')
        product_purchase_demographics_PrimeNotIncluded_net = dropout(product_purchase_demographics_PrimeNotIncluded_net, 0.25)
        product_purchase_demographics_PrimeNotIncluded_net = fully_connected(product_purchase_demographics_PrimeNotIncluded_net, 2, activation='relu')
        product_purchase_demographics_PrimeNotIncluded_net = regression(
            product_purchase_demographics_PrimeNotIncluded_net, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        tensorboard_log = MakeLogFile('', server=True)
        product_purchase_demographics_PrimeNotIncluded_model = tflearn.DNN(product_purchase_demographics_PrimeNotIncluded_net, tensorboard_dir=tensorboard_log)
    return product_purchase_demographics_PrimeNotIncluded_model, product_purchase_demographics_PrimeNotIncluded_graph

def DM_product_purchase_cm_demo_PrimeNotIncluded(run_id='', product_purchase_cm_demo_PrimeNotIncluded_graph=tf.Graph()):
    n_features = 34
    with graph.as_default():
        product_purchase_cm_demo_PrimeNotIncluded_net = input_data(shape=[None,n_features], name='{}_input'.format(run_id))
        product_purchase_cm_demo_PrimeNotIncluded_net = fully_connected(product_purchase_cm_demo_PrimeNotIncluded_net, 10, activation='relu')
        product_purchase_cm_demo_PrimeNotIncluded_net = fully_connected(product_purchase_cm_demo_PrimeNotIncluded_net, 10, activation='relu')
        product_purchase_cm_demo_PrimeNotIncluded_net = fully_connected(product_purchase_cm_demo_PrimeNotIncluded_net, 10, activation='relu')
        product_purchase_cm_demo_PrimeNotIncluded_net = dropout(product_purchase_cm_demo_PrimeNotIncluded_net, 0.25)
        product_purchase_cm_demo_PrimeNotIncluded_net = fully_connected(product_purchase_cm_demo_PrimeNotIncluded_net, 2, activation='relu')
        product_purchase_cm_demo_PrimeNotIncluded_net = regression(
            product_purchase_cm_demo_PrimeNotIncluded_net, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        tensorboard_log = MakeLogFile('', server=True)
        product_purchase_cm_demo_PrimeNotIncluded_model = tflearn.DNN(product_purchase_cm_demo_PrimeNotIncluded_net, tensorboard_dir=tensorboard_log)
    return product_purchase_cm_demo_PrimeNotIncluded_model, product_purchase_cm_demo_PrimeNotIncluded_graph

## Prime Included

def DM_product_purchase_cm_only_PrimeIncluded(run_id='', product_purchase_cm_only_PrimeIncluded_graph=tf.Graph()):
    n_features = 14
    with graph.as_default():
        product_purchase_cm_only_PrimeIncluded_net = input_data(shape=[None,n_features], name='{}_input'.format(run_id))
        product_purchase_cm_only_PrimeIncluded_net = fully_connected(product_purchase_cm_only_PrimeIncluded_net, 10, activation='relu')
        product_purchase_cm_only_PrimeIncluded_net = fully_connected(product_purchase_cm_only_PrimeIncluded_net, 10, activation='relu')
        product_purchase_cm_only_PrimeIncluded_net = fully_connected(product_purchase_cm_only_PrimeIncluded_net, 10, activation='relu')
        product_purchase_cm_only_PrimeIncluded_net = dropout(product_purchase_cm_only_PrimeIncluded_net, 0.25)
        product_purchase_cm_only_PrimeIncluded_net = fully_connected(product_purchase_cm_only_PrimeIncluded_net, 2, activation='relu')
        product_purchase_cm_only_PrimeIncluded_net = regression(
            product_purchase_cm_only_PrimeIncluded_net, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        tensorboard_log = MakeLogFile('', server=True)
        product_purchase_cm_only_PrimeIncluded_model = tflearn.DNN(product_purchase_cm_only_PrimeIncluded_net, tensorboard_dir=tensorboard_log)
    return product_purchase_cm_only_PrimeIncluded_model, product_purchase_cm_only_PrimeIncluded_graph

def DM_product_purchase_demographics_PrimeIncluded(run_id='', product_purchase_demographics_PrimeIncluded_graph=tf.Graph()):
    n_features = 27
    with graph.as_default():
        product_purchase_demographics_PrimeIncluded_net = input_data(shape=[None,n_features], name='{}_input'.format(run_id))
        product_purchase_demographics_PrimeIncluded_net = fully_connected(product_purchase_demographics_PrimeIncluded_net, 10, activation='relu')
        product_purchase_demographics_PrimeIncluded_net = fully_connected(product_purchase_demographics_PrimeIncluded_net, 10, activation='relu')
        product_purchase_demographics_PrimeIncluded_net = fully_connected(product_purchase_demographics_PrimeIncluded_net, 10, activation='relu')
        product_purchase_demographics_PrimeIncluded_net = dropout(product_purchase_demographics_PrimeIncluded_net, 0.25)
        product_purchase_demographics_PrimeIncluded_net = fully_connected(product_purchase_demographics_PrimeIncluded_net, 2, activation='relu')
        product_purchase_demographics_PrimeIncluded_net = regression(
            product_purchase_demographics_PrimeIncluded_net, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        tensorboard_log = MakeLogFile('', server=True)
        product_purchase_demographics_PrimeIncluded_model = tflearn.DNN(product_purchase_demographics_PrimeIncluded_net, tensorboard_dir=tensorboard_log)
    return product_purchase_demographics_PrimeIncluded_model, product_purchase_demographics_PrimeIncluded_graph

def DM_product_purchase_cm_demo_PrimeIncluded(run_id='', product_purchase_cm_demo_PrimeIncluded_graph=tf.Graph()):
    n_features = 41
    with graph.as_default():
        product_purchase_cm_demo_PrimeIncluded_net = input_data(shape=[None,n_features], name='{}_input'.format(run_id))
        product_purchase_cm_demo_PrimeIncluded_net = fully_connected(product_purchase_cm_demo_PrimeIncluded_net, 10, activation='relu')
        product_purchase_cm_demo_PrimeIncluded_net = fully_connected(product_purchase_cm_demo_PrimeIncluded_net, 10, activation='relu')
        product_purchase_cm_demo_PrimeIncluded_net = fully_connected(product_purchase_cm_demo_PrimeIncluded_net, 10, activation='relu')
        product_purchase_cm_demo_PrimeIncluded_net = dropout(product_purchase_cm_demo_PrimeIncluded_net, 0.25)
        product_purchase_cm_demo_PrimeIncluded_net = fully_connected(product_purchase_cm_demo_PrimeIncluded_net, 2, activation='relu')
        product_purchase_cm_demo_PrimeIncluded_net = regression(
            product_purchase_cm_demo_PrimeIncluded_net, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        tensorboard_log = MakeLogFile('', server=True)
        product_purchase_cm_demo_PrimeIncluded_model = tflearn.DNN(product_purchase_cm_demo_PrimeIncluded_net, tensorboard_dir=tensorboard_log)
    return product_purchase_cm_demo_PrimeIncluded_model, product_purchase_cm_demo_PrimeIncluded_graph

########
######## Product Based Models
#### Intent Models
## PrimeNotIncluded
def DM_product_intent_cm_only_PrimeNotIncluded(run_id='', product_intent_cm_only_PrimeNotIncluded_graph=tf.Graph()):
    n_features = 7
    with graph.as_default():
        product_intent_cm_only_PrimeNotIncluded_net = input_data(shape=[None,n_features], name='{}_input'.format(run_id))
        product_intent_cm_only_PrimeNotIncluded_net = fully_connected(product_intent_cm_only_PrimeNotIncluded_net, 10, activation='relu')
        product_intent_cm_only_PrimeNotIncluded_net = fully_connected(product_intent_cm_only_PrimeNotIncluded_net, 10, activation='relu')
        product_intent_cm_only_PrimeNotIncluded_net = fully_connected(product_intent_cm_only_PrimeNotIncluded_net, 10, activation='relu')
        product_intent_cm_only_PrimeNotIncluded_net = dropout(product_intent_cm_only_PrimeNotIncluded_net, 0.25)
        product_intent_cm_only_PrimeNotIncluded_net = fully_connected(product_intent_cm_only_PrimeNotIncluded_net, 2, activation='relu')
        product_intent_cm_only_PrimeNotIncluded_net = regression(
            product_intent_cm_only_PrimeNotIncluded_net, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        tensorboard_log = MakeLogFile('', server=True)
        product_intent_cm_only_PrimeNotIncluded_model = tflearn.DNN(product_intent_cm_only_PrimeNotIncluded_net, tensorboard_dir=tensorboard_log)
    return product_intent_cm_only_PrimeNotIncluded_model, product_intent_cm_only_PrimeNotIncluded_graph

def DM_product_intent_demographics_PrimeNotIncluded(run_id='', product_intent_demographics_PrimeNotIncluded_graph=tf.Graph()):
    n_features = 25
    with graph.as_default():
        product_intent_demographics_PrimeNotIncluded_net = input_data(shape=[None,n_features], name='{}_input'.format(run_id))
        product_intent_demographics_PrimeNotIncluded_net = fully_connected(product_intent_demographics_PrimeNotIncluded_net, 10, activation='relu')
        product_intent_demographics_PrimeNotIncluded_net = fully_connected(product_intent_demographics_PrimeNotIncluded_net, 10, activation='relu')
        product_intent_demographics_PrimeNotIncluded_net = fully_connected(product_intent_demographics_PrimeNotIncluded_net, 10, activation='relu')
        product_intent_demographics_PrimeNotIncluded_net = dropout(product_intent_demographics_PrimeNotIncluded_net, 0.25)
        product_intent_demographics_PrimeNotIncluded_net = fully_connected(product_intent_demographics_PrimeNotIncluded_net, 2, activation='relu')
        product_intent_demographics_PrimeNotIncluded_net = regression(
            product_intent_demographics_PrimeNotIncluded_net, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        tensorboard_log = MakeLogFile('', server=True)
        product_intent_demographics_PrimeNotIncluded_model = tflearn.DNN(product_intent_demographics_PrimeNotIncluded_net, tensorboard_dir=tensorboard_log)
    return product_intent_demographics_PrimeNotIncluded_model, product_intent_demographics_PrimeNotIncluded_graph

def DM_product_intent_cm_demo_PrimeNotIncluded(run_id='', product_intent_cm_demo_PrimeNotIncluded_graph=tf.Graph()):
    n_features = 32
    with graph.as_default():
        product_intent_cm_demo_PrimeNotIncluded_net = input_data(shape=[None,n_features], name='{}_input'.format(run_id))
        product_intent_cm_demo_PrimeNotIncluded_net = fully_connected(product_intent_cm_demo_PrimeNotIncluded_net, 10, activation='relu')
        product_intent_cm_demo_PrimeNotIncluded_net = fully_connected(product_intent_cm_demo_PrimeNotIncluded_net, 10, activation='relu')
        product_intent_cm_demo_PrimeNotIncluded_net = fully_connected(product_intent_cm_demo_PrimeNotIncluded_net, 10, activation='relu')
        product_intent_cm_demo_PrimeNotIncluded_net = dropout(product_intent_cm_demo_PrimeNotIncluded_net, 0.25)
        product_intent_cm_demo_PrimeNotIncluded_net = fully_connected(product_intent_cm_demo_PrimeNotIncluded_net, 2, activation='relu')
        product_intent_cm_demo_PrimeNotIncluded_net = regression(
            product_intent_cm_demo_PrimeNotIncluded_net, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        tensorboard_log = MakeLogFile('', server=True)
        product_intent_cm_demo_PrimeNotIncluded_model = tflearn.DNN(product_intent_cm_demo_PrimeNotIncluded_net, tensorboard_dir=tensorboard_log)
    return product_intent_cm_demo_PrimeNotIncluded_model, product_intent_cm_demo_PrimeNotIncluded_graph

## Prime Included

def DM_product_intent_cm_only_PrimeIncluded(run_id='', product_intent_cm_only_PrimeIncluded_graph=tf.Graph()):
    n_features = 14
    with graph.as_default():
        product_intent_cm_only_PrimeIncluded_net = input_data(shape=[None,n_features], name='{}_input'.format(run_id))
        product_intent_cm_only_PrimeIncluded_net = fully_connected(product_intent_cm_only_PrimeIncluded_net, 10, activation='relu')
        product_intent_cm_only_PrimeIncluded_net = fully_connected(product_intent_cm_only_PrimeIncluded_net, 10, activation='relu')
        product_intent_cm_only_PrimeIncluded_net = fully_connected(product_intent_cm_only_PrimeIncluded_net, 10, activation='relu')
        product_intent_cm_only_PrimeIncluded_net = dropout(product_intent_cm_only_PrimeIncluded_net, 0.25)
        product_intent_cm_only_PrimeIncluded_net = fully_connected(product_intent_cm_only_PrimeIncluded_net, 2, activation='relu')
        product_intent_cm_only_PrimeIncluded_net = regression(
            product_intent_cm_only_PrimeIncluded_net, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        tensorboard_log = MakeLogFile('', server=True)
        product_intent_cm_only_PrimeIncluded_model = tflearn.DNN(product_intent_cm_only_PrimeIncluded_net, tensorboard_dir=tensorboard_log)
    return product_intent_cm_only_PrimeIncluded_model, product_intent_cm_only_PrimeIncluded_graph

def DM_product_intent_demographics_PrimeIncluded(run_id='', product_intent_demographics_PrimeIncluded_graph=tf.Graph()):
    n_features = 25
    with graph.as_default():
        product_intent_demographics_PrimeIncluded_net = input_data(shape=[None,n_features], name='{}_input'.format(run_id))
        product_intent_demographics_PrimeIncluded_net = fully_connected(product_intent_demographics_PrimeIncluded_net, 10, activation='relu')
        product_intent_demographics_PrimeIncluded_net = fully_connected(product_intent_demographics_PrimeIncluded_net, 10, activation='relu')
        product_intent_demographics_PrimeIncluded_net = fully_connected(product_intent_demographics_PrimeIncluded_net, 10, activation='relu')
        product_intent_demographics_PrimeIncluded_net = dropout(product_intent_demographics_PrimeIncluded_net, 0.25)
        product_intent_demographics_PrimeIncluded_net = fully_connected(product_intent_demographics_PrimeIncluded_net, 2, activation='relu')
        product_intent_demographics_PrimeIncluded_net = regression(
            product_intent_demographics_PrimeIncluded_net, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        tensorboard_log = MakeLogFile('', server=True)
        product_intent_demographics_PrimeIncluded_model = tflearn.DNN(product_intent_demographics_PrimeIncluded_net, tensorboard_dir=tensorboard_log)
    return product_intent_demographics_PrimeIncluded_model, product_intent_demographics_PrimeIncluded_graph

def DM_product_intent_cm_demo_PrimeIncluded(run_id='', product_intent_cm_demo_PrimeIncluded_graph=tf.Graph()):
    n_features = 39
    with graph.as_default():
        product_intent_cm_demo_PrimeIncluded_net = input_data(shape=[None,n_features], name='{}_input'.format(run_id))
        product_intent_cm_demo_PrimeIncluded_net = fully_connected(product_intent_cm_demo_PrimeIncluded_net, 10, activation='relu')
        product_intent_cm_demo_PrimeIncluded_net = fully_connected(product_intent_cm_demo_PrimeIncluded_net, 10, activation='relu')
        product_intent_cm_demo_PrimeIncluded_net = fully_connected(product_intent_cm_demo_PrimeIncluded_net, 10, activation='relu')
        product_intent_cm_demo_PrimeIncluded_net = dropout(product_intent_cm_demo_PrimeIncluded_net, 0.25)
        product_intent_cm_demo_PrimeIncluded_net = fully_connected(product_intent_cm_demo_PrimeIncluded_net, 2, activation='relu')
        product_intent_cm_demo_PrimeIncluded_net = regression(
            product_intent_cm_demo_PrimeIncluded_net, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        tensorboard_log = MakeLogFile('', server=True)
        product_intent_cm_demo_PrimeIncluded_model = tflearn.DNN(product_intent_cm_demo_PrimeIncluded_net, tensorboard_dir=tensorboard_log)
    return product_intent_cm_demo_PrimeIncluded_model, product_intent_cm_demo_PrimeIncluded_graph


###################
########
######## User Based Model
#### Purchase Models
## PrimeNotIncluded
def DM_user_purchase_cm_only_PrimeNotIncluded(run_id='', user_purchase_cm_only_PrimeNotIncluded_graph=tf.Graph()):
    n_features = 7
    with graph.as_default():
        user_purchase_cm_only_PrimeNotIncluded_net = input_data(shape=[None,n_features], name='{}_input'.format(run_id))
        user_purchase_cm_only_PrimeNotIncluded_net = fully_connected(user_purchase_cm_only_PrimeNotIncluded_net, 10, activation='relu')
        user_purchase_cm_only_PrimeNotIncluded_net = fully_connected(user_purchase_cm_only_PrimeNotIncluded_net, 10, activation='relu')
        user_purchase_cm_only_PrimeNotIncluded_net = fully_connected(user_purchase_cm_only_PrimeNotIncluded_net, 10, activation='relu')
        user_purchase_cm_only_PrimeNotIncluded_net = dropout(user_purchase_cm_only_PrimeNotIncluded_net, 0.25)
        user_purchase_cm_only_PrimeNotIncluded_net = fully_connected(user_purchase_cm_only_PrimeNotIncluded_net, 2, activation='relu')
        user_purchase_cm_only_PrimeNotIncluded_net = regression(
            user_purchase_cm_only_PrimeNotIncluded_net, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        tensorboard_log = MakeLogFile('', server=True)
        user_purchase_cm_only_PrimeNotIncluded_model = tflearn.DNN(user_purchase_cm_only_PrimeNotIncluded_net, tensorboard_dir=tensorboard_log)
    return user_purchase_cm_only_PrimeNotIncluded_model, user_purchase_cm_only_PrimeNotIncluded_graph

def DM_user_purchase_demographics_PrimeNotIncluded(run_id='', user_purchase_demographics_PrimeNotIncluded_graph=tf.Graph()):
    n_features = 27
    with graph.as_default():
        user_purchase_demographics_PrimeNotIncluded_net = input_data(shape=[None,n_features], name='{}_input'.format(run_id))
        user_purchase_demographics_PrimeNotIncluded_net = fully_connected(user_purchase_demographics_PrimeNotIncluded_net, 10, activation='relu')
        user_purchase_demographics_PrimeNotIncluded_net = fully_connected(user_purchase_demographics_PrimeNotIncluded_net, 10, activation='relu')
        user_purchase_demographics_PrimeNotIncluded_net = fully_connected(user_purchase_demographics_PrimeNotIncluded_net, 10, activation='relu')
        user_purchase_demographics_PrimeNotIncluded_net = dropout(user_purchase_demographics_PrimeNotIncluded_net, 0.25)
        user_purchase_demographics_PrimeNotIncluded_net = fully_connected(user_purchase_demographics_PrimeNotIncluded_net, 2, activation='relu')
        user_purchase_demographics_PrimeNotIncluded_net = regression(
            user_purchase_demographics_PrimeNotIncluded_net, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        tensorboard_log = MakeLogFile('', server=True)
        user_purchase_demographics_PrimeNotIncluded_model = tflearn.DNN(user_purchase_demographics_PrimeNotIncluded_net, tensorboard_dir=tensorboard_log)
    return user_purchase_demographics_PrimeNotIncluded_model, user_purchase_demographics_PrimeNotIncluded_graph

def DM_user_purchase_cm_demo_PrimeNotIncluded(run_id='', user_purchase_cm_demo_PrimeNotIncluded_graph=tf.Graph()):
    n_features = 34
    with graph.as_default():
        user_purchase_cm_demo_PrimeNotIncluded_net = input_data(shape=[None,n_features], name='{}_input'.format(run_id))
        user_purchase_cm_demo_PrimeNotIncluded_net = fully_connected(user_purchase_cm_demo_PrimeNotIncluded_net, 10, activation='relu')
        user_purchase_cm_demo_PrimeNotIncluded_net = fully_connected(user_purchase_cm_demo_PrimeNotIncluded_net, 10, activation='relu')
        user_purchase_cm_demo_PrimeNotIncluded_net = fully_connected(user_purchase_cm_demo_PrimeNotIncluded_net, 10, activation='relu')
        user_purchase_cm_demo_PrimeNotIncluded_net = dropout(user_purchase_cm_demo_PrimeNotIncluded_net, 0.25)
        user_purchase_cm_demo_PrimeNotIncluded_net = fully_connected(user_purchase_cm_demo_PrimeNotIncluded_net, 2, activation='relu')
        user_purchase_cm_demo_PrimeNotIncluded_net = regression(
            user_purchase_cm_demo_PrimeNotIncluded_net, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        tensorboard_log = MakeLogFile('', server=True)
        user_purchase_cm_demo_PrimeNotIncluded_model = tflearn.DNN(user_purchase_cm_demo_PrimeNotIncluded_net, tensorboard_dir=tensorboard_log)
    return user_purchase_cm_demo_PrimeNotIncluded_model, user_purchase_cm_demo_PrimeNotIncluded_graph

## Prime Included

def DM_user_purchase_cm_only_PrimeIncluded(run_id='', user_purchase_cm_only_PrimeIncluded_graph=tf.Graph()):
    n_features = 14
    with graph.as_default():
        user_purchase_cm_only_PrimeIncluded_net = input_data(shape=[None,n_features], name='{}_input'.format(run_id))
        user_purchase_cm_only_PrimeIncluded_net = fully_connected(user_purchase_cm_only_PrimeIncluded_net, 10, activation='relu')
        user_purchase_cm_only_PrimeIncluded_net = fully_connected(user_purchase_cm_only_PrimeIncluded_net, 10, activation='relu')
        user_purchase_cm_only_PrimeIncluded_net = fully_connected(user_purchase_cm_only_PrimeIncluded_net, 10, activation='relu')
        user_purchase_cm_only_PrimeIncluded_net = dropout(user_purchase_cm_only_PrimeIncluded_net, 0.25)
        user_purchase_cm_only_PrimeIncluded_net = fully_connected(user_purchase_cm_only_PrimeIncluded_net, 2, activation='relu')
        user_purchase_cm_only_PrimeIncluded_net = regression(
            user_purchase_cm_only_PrimeIncluded_net, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        tensorboard_log = MakeLogFile('', server=True)
        user_purchase_cm_only_PrimeIncluded_model = tflearn.DNN(user_purchase_cm_only_PrimeIncluded_net, tensorboard_dir=tensorboard_log)
    return user_purchase_cm_only_PrimeIncluded_model, user_purchase_cm_only_PrimeIncluded_graph

def DM_user_purchase_demographics_PrimeIncluded(run_id='', user_purchase_demographics_PrimeIncluded_graph=tf.Graph()):
    n_features = 27
    with graph.as_default():
        user_purchase_demographics_PrimeIncluded_net = input_data(shape=[None,n_features], name='{}_input'.format(run_id))
        user_purchase_demographics_PrimeIncluded_net = fully_connected(user_purchase_demographics_PrimeIncluded_net, 10, activation='relu')
        user_purchase_demographics_PrimeIncluded_net = fully_connected(user_purchase_demographics_PrimeIncluded_net, 10, activation='relu')
        user_purchase_demographics_PrimeIncluded_net = fully_connected(user_purchase_demographics_PrimeIncluded_net, 10, activation='relu')
        user_purchase_demographics_PrimeIncluded_net = dropout(user_purchase_demographics_PrimeIncluded_net, 0.25)
        user_purchase_demographics_PrimeIncluded_net = fully_connected(user_purchase_demographics_PrimeIncluded_net, 2, activation='relu')
        user_purchase_demographics_PrimeIncluded_net = regression(
            user_purchase_demographics_PrimeIncluded_net, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        tensorboard_log = MakeLogFile('', server=True)
        user_purchase_demographics_PrimeIncluded_model = tflearn.DNN(user_purchase_demographics_PrimeIncluded_net, tensorboard_dir=tensorboard_log)
    return user_purchase_demographics_PrimeIncluded_model, user_purchase_demographics_PrimeIncluded_graph

def DM_user_purchase_cm_demo_PrimeIncluded(run_id='', user_purchase_cm_demo_PrimeIncluded_graph=tf.Graph()):
    n_features = 41
    with graph.as_default():
        user_purchase_cm_demo_PrimeIncluded_net = input_data(shape=[None,n_features], name='{}_input'.format(run_id))
        user_purchase_cm_demo_PrimeIncluded_net = fully_connected(user_purchase_cm_demo_PrimeIncluded_net, 10, activation='relu')
        user_purchase_cm_demo_PrimeIncluded_net = fully_connected(user_purchase_cm_demo_PrimeIncluded_net, 10, activation='relu')
        user_purchase_cm_demo_PrimeIncluded_net = fully_connected(user_purchase_cm_demo_PrimeIncluded_net, 10, activation='relu')
        user_purchase_cm_demo_PrimeIncluded_net = dropout(user_purchase_cm_demo_PrimeIncluded_net, 0.25)
        user_purchase_cm_demo_PrimeIncluded_net = fully_connected(user_purchase_cm_demo_PrimeIncluded_net, 2, activation='relu')
        user_purchase_cm_demo_PrimeIncluded_net = regression(
            user_purchase_cm_demo_PrimeIncluded_net, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        tensorboard_log = MakeLogFile('', server=True)
        user_purchase_cm_demo_PrimeIncluded_model = tflearn.DNN(user_purchase_cm_demo_PrimeIncluded_net, tensorboard_dir=tensorboard_log)
    return user_purchase_cm_demo_PrimeIncluded_model, user_purchase_cm_demo_PrimeIncluded_graph

########
######## Product Based Models
#### Intent Models
## PrimeNotIncluded
def DM_user_intent_cm_only_PrimeNotIncluded(run_id='', user_intent_cm_only_PrimeNotIncluded_graph=tf.Graph()):
    n_features = 7
    with graph.as_default():
        user_intent_cm_only_PrimeNotIncluded_net = input_data(shape=[None,n_features], name='{}_input'.format(run_id))
        user_intent_cm_only_PrimeNotIncluded_net = fully_connected(user_intent_cm_only_PrimeNotIncluded_net, 10, activation='relu')
        user_intent_cm_only_PrimeNotIncluded_net = fully_connected(user_intent_cm_only_PrimeNotIncluded_net, 10, activation='relu')
        user_intent_cm_only_PrimeNotIncluded_net = fully_connected(user_intent_cm_only_PrimeNotIncluded_net, 10, activation='relu')
        user_intent_cm_only_PrimeNotIncluded_net = dropout(user_intent_cm_only_PrimeNotIncluded_net, 0.25)
        user_intent_cm_only_PrimeNotIncluded_net = fully_connected(user_intent_cm_only_PrimeNotIncluded_net, 2, activation='relu')
        user_intent_cm_only_PrimeNotIncluded_net = regression(
            user_intent_cm_only_PrimeNotIncluded_net, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        tensorboard_log = MakeLogFile('', server=True)
        user_intent_cm_only_PrimeNotIncluded_model = tflearn.DNN(user_intent_cm_only_PrimeNotIncluded_net, tensorboard_dir=tensorboard_log)
    return user_intent_cm_only_PrimeNotIncluded_model, user_intent_cm_only_PrimeNotIncluded_graph

def DM_user_intent_demographics_PrimeNotIncluded(run_id='', user_intent_demographics_PrimeNotIncluded_graph=tf.Graph()):
    n_features = 25
    with graph.as_default():
        user_intent_demographics_PrimeNotIncluded_net = input_data(shape=[None,n_features], name='{}_input'.format(run_id))
        user_intent_demographics_PrimeNotIncluded_net = fully_connected(user_intent_demographics_PrimeNotIncluded_net, 10, activation='relu')
        user_intent_demographics_PrimeNotIncluded_net = fully_connected(user_intent_demographics_PrimeNotIncluded_net, 10, activation='relu')
        user_intent_demographics_PrimeNotIncluded_net = fully_connected(user_intent_demographics_PrimeNotIncluded_net, 10, activation='relu')
        user_intent_demographics_PrimeNotIncluded_net = dropout(user_intent_demographics_PrimeNotIncluded_net, 0.25)
        user_intent_demographics_PrimeNotIncluded_net = fully_connected(user_intent_demographics_PrimeNotIncluded_net, 2, activation='relu')
        user_intent_demographics_PrimeNotIncluded_net = regression(
            user_intent_demographics_PrimeNotIncluded_net, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        tensorboard_log = MakeLogFile('', server=True)
        user_intent_demographics_PrimeNotIncluded_model = tflearn.DNN(user_intent_demographics_PrimeNotIncluded_net, tensorboard_dir=tensorboard_log)
    return user_intent_demographics_PrimeNotIncluded_model, user_intent_demographics_PrimeNotIncluded_graph

def DM_user_intent_cm_demo_PrimeNotIncluded(run_id='', user_intent_cm_demo_PrimeNotIncluded_graph=tf.Graph()):
    n_features = 32
    with graph.as_default():
        user_intent_cm_demo_PrimeNotIncluded_net = input_data(shape=[None,n_features], name='{}_input'.format(run_id))
        user_intent_cm_demo_PrimeNotIncluded_net = fully_connected(user_intent_cm_demo_PrimeNotIncluded_net, 10, activation='relu')
        user_intent_cm_demo_PrimeNotIncluded_net = fully_connected(user_intent_cm_demo_PrimeNotIncluded_net, 10, activation='relu')
        user_intent_cm_demo_PrimeNotIncluded_net = fully_connected(user_intent_cm_demo_PrimeNotIncluded_net, 10, activation='relu')
        user_intent_cm_demo_PrimeNotIncluded_net = dropout(user_intent_cm_demo_PrimeNotIncluded_net, 0.25)
        user_intent_cm_demo_PrimeNotIncluded_net = fully_connected(user_intent_cm_demo_PrimeNotIncluded_net, 2, activation='relu')
        user_intent_cm_demo_PrimeNotIncluded_net = regression(
            user_intent_cm_demo_PrimeNotIncluded_net, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        tensorboard_log = MakeLogFile('', server=True)
        user_intent_cm_demo_PrimeNotIncluded_model = tflearn.DNN(user_intent_cm_demo_PrimeNotIncluded_net, tensorboard_dir=tensorboard_log)
    return user_intent_cm_demo_PrimeNotIncluded_model, user_intent_cm_demo_PrimeNotIncluded_graph

## Prime Included

def DM_user_intent_cm_only_PrimeIncluded(run_id='', user_intent_cm_only_PrimeIncluded_graph=tf.Graph()):
    n_features = 14
    with graph.as_default():
        user_intent_cm_only_PrimeIncluded_net = input_data(shape=[None,n_features], name='{}_input'.format(run_id))
        user_intent_cm_only_PrimeIncluded_net = fully_connected(user_intent_cm_only_PrimeIncluded_net, 10, activation='relu')
        user_intent_cm_only_PrimeIncluded_net = fully_connected(user_intent_cm_only_PrimeIncluded_net, 10, activation='relu')
        user_intent_cm_only_PrimeIncluded_net = fully_connected(user_intent_cm_only_PrimeIncluded_net, 10, activation='relu')
        user_intent_cm_only_PrimeIncluded_net = dropout(user_intent_cm_only_PrimeIncluded_net, 0.25)
        user_intent_cm_only_PrimeIncluded_net = fully_connected(user_intent_cm_only_PrimeIncluded_net, 2, activation='relu')
        user_intent_cm_only_PrimeIncluded_net = regression(
            user_intent_cm_only_PrimeIncluded_net, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        tensorboard_log = MakeLogFile('', server=True)
        user_intent_cm_only_PrimeIncluded_model = tflearn.DNN(user_intent_cm_only_PrimeIncluded_net, tensorboard_dir=tensorboard_log)
    return user_intent_cm_only_PrimeIncluded_model, user_intent_cm_only_PrimeIncluded_graph

def DM_user_intent_demographics_PrimeIncluded(run_id='', user_intent_demographics_PrimeIncluded_graph=tf.Graph()):
    n_features = 25
    with graph.as_default():
        user_intent_demographics_PrimeIncluded_net = input_data(shape=[None,n_features], name='{}_input'.format(run_id))
        user_intent_demographics_PrimeIncluded_net = fully_connected(user_intent_demographics_PrimeIncluded_net, 10, activation='relu')
        user_intent_demographics_PrimeIncluded_net = fully_connected(user_intent_demographics_PrimeIncluded_net, 10, activation='relu')
        user_intent_demographics_PrimeIncluded_net = fully_connected(user_intent_demographics_PrimeIncluded_net, 10, activation='relu')
        user_intent_demographics_PrimeIncluded_net = dropout(user_intent_demographics_PrimeIncluded_net, 0.25)
        user_intent_demographics_PrimeIncluded_net = fully_connected(user_intent_demographics_PrimeIncluded_net, 2, activation='relu')
        user_intent_demographics_PrimeIncluded_net = regression(
            user_intent_demographics_PrimeIncluded_net, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        tensorboard_log = MakeLogFile('', server=True)
        user_intent_demographics_PrimeIncluded_model = tflearn.DNN(user_intent_demographics_PrimeIncluded_net, tensorboard_dir=tensorboard_log)
    return user_intent_demographics_PrimeIncluded_model, user_intent_demographics_PrimeIncluded_graph

def DM_user_intent_cm_demo_PrimeIncluded(run_id='', user_intent_cm_demo_PrimeIncluded_graph=tf.Graph()):
    n_features = 39
    with graph.as_default():
        user_intent_cm_demo_PrimeIncluded_net = input_data(shape=[None,n_features], name='{}_input'.format(run_id))
        user_intent_cm_demo_PrimeIncluded_net = fully_connected(user_intent_cm_demo_PrimeIncluded_net, 10, activation='relu')
        user_intent_cm_demo_PrimeIncluded_net = fully_connected(user_intent_cm_demo_PrimeIncluded_net, 10, activation='relu')
        user_intent_cm_demo_PrimeIncluded_net = fully_connected(user_intent_cm_demo_PrimeIncluded_net, 10, activation='relu')
        user_intent_cm_demo_PrimeIncluded_net = dropout(user_intent_cm_demo_PrimeIncluded_net, 0.25)
        user_intent_cm_demo_PrimeIncluded_net = fully_connected(user_intent_cm_demo_PrimeIncluded_net, 2, activation='relu')
        user_intent_cm_demo_PrimeIncluded_net = regression(
            user_intent_cm_demo_PrimeIncluded_net, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        tensorboard_log = MakeLogFile('', server=True)
        user_intent_cm_demo_PrimeIncluded_model = tflearn.DNN(user_intent_cm_demo_PrimeIncluded_net, tensorboard_dir=tensorboard_log)
    return user_intent_cm_demo_PrimeIncluded_model, user_intent_cm_demo_PrimeIncluded_graph

###################
###################
###################

def ReadyData(data, test_size, do_shuffle=True):
    if do_shuffle:
        shuffle_while = True
        while shuffle_while:
            numpy.random.shuffle(data)
            train_data = data[:-test_size]
            test_data = data[-test_size:]
            X,Y = zip(*train_data)
            test_x, test_y = zip(*test_data)
            Y = list(Y)
            rate_0 = Y.count([1,0])/len(Y)
            rate_1 = Y.count([0,1])/len(Y)
            if rate_0>0.02 and rate_1>0.02:
                shuffle_while = False
            X = numpy.array(list(X))#,dtype=object
            Y = numpy.array(list(Y))
            test_x = numpy.array(list(test_x))#,dtype=object
            test_y = numpy.array(list(test_y)) 
    else:
        train_data = data[:-test_size]
        test_data = data[-test_size:]
        X,Y = zip(*train_data)
        test_x, test_y = zip(*test_data)
        X = numpy.array(list(X))#,dtype=object
        Y = numpy.array(list(Y))
        test_x = numpy.array(list(test_x))#,dtype=object
        test_y = numpy.array(list(test_y))
    return X,Y,test_x,test_y

def LoadData_product(product_id, category=2, target='purchase', mltype='cm_only', prime_inclusion=False, do_shuffle=True, test_size=600, server=True):
    x = getXvector_product(product_id, target=target,mltype=mltype, prime_inclusion=prime_inclusion, server=server)
    if target == 'purchase':
        y = getYlabel_product_purchase(product_id, category=category, server=server)
    if target == 'intent':
        y = getYlabel_product_intent(product_id, category=category, server=server)
    y = [[0,1] if val==1 else [1,0] for val in y]
    # x = numpy.array(x)
    # y = numpy.array(y)
    data = numpy.array([list(val) for val in zip(x,y)])
    X,Y,test_x,test_y = ReadyData(data, test_size=test_size, do_shuffle=True)
    return X,Y,test_x,test_y

def LoadData_user(user_id, category=2, target='purchase', mltype='cm_only', prime_inclusion=False, do_shuffle=True, test_size=7, server=True):
    x = getXvector_user(user_id, target=target,mltype=mltype, prime_inclusion=prime_inclusion, server=server)
    if target == 'purchase':
        y = getYlabel_user_purchase(user_id, category=category, server=server)
    if target == 'intent':
        y = getYlabel_user_intent(user_id, category=category, server=server)
    y = [[0,1] if val==1 else [1,0] for val in y]
    # x = numpy.array(x)
    # y = numpy.array(y)
    data = numpy.array([list(val) for val in zip(x,y)])
    X,Y,test_x,test_y = ReadyData(data, test_size=test_size, do_shuffle=True)
    return X,Y,test_x,test_y

def getMetrics(preds, test_y):
    y_pred = [pred.argmax() for pred in preds]
    processed_test_y = [ty.argmax() for ty in test_y]
    TP = sum([1 if (y_pred[i]+y) == 2 else 0 for i,y in enumerate(processed_test_y)])
    TN = sum([1 if (y_pred[i]+y) == 0 else 0 for i,y in enumerate(processed_test_y)])
    FP = sum([1 if (y_pred[i]==1 and y == 0) else 0 for i,y in enumerate(processed_test_y)])
    FN = sum([1 if (y_pred[i]==0 and y == 1) else 0 for i,y in enumerate(processed_test_y)])
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, accuracy, f1

###################
###################
###################
###################
###################
###################

def Train(base_id, category=2, mode='product', target='purchase', mltype='cm_only', prime_inclusion=False):
    # Train any amount of 10x epochs so it gets to at least 15 mins error #
    tf.reset_default_graph()
    prime_inclusion_str = "PrimeIncluded" if prime_inclusion else "PrimeNotIncluded"
    version = 1
    times_run = 1
    if mode=='product':
        product_id = base_id
        model_name = 'NomuraSoken_{}_{}_{}__{}__v{}'.format(mode, target, mltype, prime_inclusion_str, product_id, version)
        log_file = MakeLogFile('NeuralNetworkTests_product.csv',server=True)
    if mode=='user':
        user_id = base_id
        model_name = 'NomuraSoken_{}_{}_{}__{}__v{}'.format(mode, target, mltype, prime_inclusion_str, user_id, version)
        log_file = MakeLogFile('NeuralNetworkTests_user.csv',server=True)
    run_id = '{}_run{}'.format(model_name,times_run)
    while os.path.exists(os.path.join(MakeLogFile('', server=True), run_id)):
        times_run += 1
        run_id = '{}_run{}'.format(model_name,times_run)
    model_path = os.path.abspath(MakeModelPath('',server=True)+'/neural_network/{0}/{0}.tfl'.format(run_id))
    model_dir = os.path.abspath(MakeModelPath('',server=True)+'/neural_network/{0}'.format(run_id))
    ######################
    if mode == 'product':
        if target == 'purchase':
            if prime_inclusion == False:
                if mltype == 'cm_only':
                    X,Y,test_x,test_y = LoadData_product(product_id,category,target,mltype,prime_inclusion)
                    product_purchase_cm_only_PrimeNotIncluded_model, product_purchase_cm_only_PrimeNotIncluded_graph = DM_product_purchase_cm_only_PrimeNotIncluded(run_id,tf.Graph())
                    with product_purchase_cm_only_PrimeNotIncluded_graph.as_default():
                        product_purchase_cm_only_PrimeNotIncluded_model.fit(
                            {'{}_input'.format(run_id): X},
                            {'{}_targets'.format(run_id): Y},
                            n_epoch=5,
                            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                            show_metric=True,
                            batch_size=None,
                            shuffle=False,
                            snapshot_step=None,
                            run_id=run_id
                            )
                        product_purchase_cm_only_PrimeNotIncluded_model.save(model_path)
                        preds = product_purchase_cm_only_PrimeNotIncluded_model.predict(test_x)
                        precision, recall, accuracy, f1 = getMetrics(preds, test_y)
                        strlog = "{},{},{},{},{},{},{},{},{},{}".format(mode,target,mltype,prime_inclusion_str,product_id,version,precision,recall,accuracy,f1)
                        printLog(strlog,log_file)
                if mltype == 'demographics':
                    X,Y,test_x,test_y = LoadData_product(product_id,category,target,mltype,prime_inclusion)
                    product_purchase_demographics_PrimeNotIncluded_model, product_purchase_demographics_PrimeNotIncluded_graph = DM_product_purchase_demographics_PrimeNotIncluded(run_id,tf.Graph())
                    with product_purchase_demographics_PrimeNotIncluded_graph.as_default():
                        product_purchase_demographics_PrimeNotIncluded_model.fit(
                            {'{}_input'.format(run_id): X},
                            {'{}_targets'.format(run_id): Y},
                            n_epoch=5,
                            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                            show_metric=True,
                            batch_size=None,
                            shuffle=False,
                            snapshot_step=None,
                            run_id=run_id
                            )
                        product_purchase_demographics_PrimeNotIncluded_model.save(model_path)
                        preds = product_purchase_demographics_PrimeNotIncluded_model.predict(test_x)
                        precision, recall, accuracy, f1 = getMetrics(preds, test_y)
                        strlog = "{},{},{},{},{},{},{},{},{},{}".format(mode,target,mltype,prime_inclusion_str,product_id,version,precision,recall,accuracy,f1)
                        printLog(strlog,log_file)
                if mltype == 'cm_demo':
                    X,Y,test_x,test_y = LoadData_product(product_id,category,target,mltype,prime_inclusion)
                    product_purchase_cm_demo_PrimeNotIncluded_model, product_purchase_cm_demo_PrimeNotIncluded_graph = DM_product_purchase_cm_demo_PrimeNotIncluded(run_id,tf.Graph())
                    with product_purchase_cm_demo_PrimeNotIncluded_graph.as_default():
                        product_purchase_cm_demo_PrimeNotIncluded_model.fit(
                            {'{}_input'.format(run_id): X},
                            {'{}_targets'.format(run_id): Y},
                            n_epoch=5,
                            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                            show_metric=True,
                            batch_size=None,
                            shuffle=False,
                            snapshot_step=None,
                            run_id=run_id
                            )
                        product_purchase_cm_demo_PrimeNotIncluded_model.save(model_path)
                        preds = product_purchase_cm_demo_PrimeNotIncluded_model.predict(test_x)
                        precision, recall, accuracy, f1 = getMetrics(preds, test_y)
                        strlog = "{},{},{},{},{},{},{},{},{},{}".format(mode,target,mltype,prime_inclusion_str,product_id,version,precision,recall,accuracy,f1)
                        printLog(strlog,log_file)
            if prime_inclusion == True:
                if mltype == 'cm_only':
                    X,Y,test_x,test_y = LoadData_product(product_id,category,target,mltype,prime_inclusion)
                    product_purchase_cm_only_PrimeIncluded_model, product_purchase_cm_only_PrimeIncluded_graph = DM_product_purchase_cm_only_PrimeIncluded(run_id,tf.Graph())
                    with product_purchase_cm_only_PrimeIncluded_graph.as_default():
                        product_purchase_cm_only_PrimeIncluded_model.fit(
                            {'{}_input'.format(run_id): X},
                            {'{}_targets'.format(run_id): Y},
                            n_epoch=5,
                            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                            show_metric=True,
                            batch_size=None,
                            shuffle=False,
                            snapshot_step=None,
                            run_id=run_id
                            )
                        product_purchase_cm_only_PrimeIncluded_model.save(model_path)
                        preds = product_purchase_cm_only_PrimeIncluded_model.predict(test_x)
                        precision, recall, accuracy, f1 = getMetrics(preds, test_y)
                        strlog = "{},{},{},{},{},{},{},{},{},{}".format(mode,target,mltype,prime_inclusion_str,product_id,version,precision,recall,accuracy,f1)
                        printLog(strlog,log_file)
                if mltype == 'demographics':
                    X,Y,test_x,test_y = LoadData_product(product_id,category,target,mltype,prime_inclusion)
                    product_purchase_demographics_PrimeIncluded_model, product_purchase_demographics_PrimeIncluded_graph = DM_product_purchase_demographics_PrimeIncluded(run_id,tf.Graph())
                    with product_purchase_demographics_PrimeIncluded_graph.as_default():
                        product_purchase_demographics_PrimeIncluded_model.fit(
                            {'{}_input'.format(run_id): X},
                            {'{}_targets'.format(run_id): Y},
                            n_epoch=5,
                            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                            show_metric=True,
                            batch_size=None,
                            shuffle=False,
                            snapshot_step=None,
                            run_id=run_id
                            )
                        product_purchase_demographics_PrimeIncluded_model.save(model_path)
                        preds = product_purchase_demographics_PrimeIncluded_model.predict(test_x)
                        precision, recall, accuracy, f1 = getMetrics(preds, test_y)
                        strlog = "{},{},{},{},{},{},{},{},{},{}".format(mode,target,mltype,prime_inclusion_str,product_id,version,precision,recall,accuracy,f1)
                        printLog(strlog,log_file)
                if mltype == 'cm_demo':
                    X,Y,test_x,test_y = LoadData_product(product_id,category,target,mltype,prime_inclusion)
                    product_purchase_cm_demo_PrimeIncluded_model, product_purchase_cm_demo_PrimeIncluded_graph = DM_product_purchase_cm_demo_PrimeIncluded(run_id,tf.Graph())
                    with product_purchase_cm_demo_PrimeIncluded_graph.as_default():
                        product_purchase_cm_demo_PrimeIncluded_model.fit(
                            {'{}_input'.format(run_id): X},
                            {'{}_targets'.format(run_id): Y},
                            n_epoch=5,
                            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                            show_metric=True,
                            batch_size=None,
                            shuffle=False,
                            snapshot_step=None,
                            run_id=run_id
                            )
                        product_purchase_cm_demo_PrimeIncluded_model.save(model_path)
                        preds = product_purchase_cm_demo_PrimeIncluded_model.predict(test_x)
                        precision, recall, accuracy, f1 = getMetrics(preds, test_y)
                        strlog = "{},{},{},{},{},{},{},{},{},{}".format(mode,target,mltype,prime_inclusion_str,product_id,version,precision,recall,accuracy,f1)
                        printLog(strlog,log_file)
        if target == 'intent':
            if prime_inclusion == False:
                if mltype == 'cm_only':
                    X,Y,test_x,test_y = LoadData_product(product_id,category,target,mltype,prime_inclusion)
                    product_intent_cm_only_PrimeNotIncluded_model, product_intent_cm_only_PrimeNotIncluded_graph = DM_product_intent_cm_only_PrimeNotIncluded(run_id,tf.Graph())
                    with product_intent_cm_only_PrimeNotIncluded_graph.as_default():
                        product_intent_cm_only_PrimeNotIncluded_model.fit(
                            {'{}_input'.format(run_id): X},
                            {'{}_targets'.format(run_id): Y},
                            n_epoch=5,
                            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                            show_metric=True,
                            batch_size=None,
                            shuffle=False,
                            snapshot_step=None,
                            run_id=run_id
                            )
                        product_intent_cm_only_PrimeNotIncluded_model.save(model_path)
                        preds = product_intent_cm_only_PrimeNotIncluded_model.predict(test_x)
                        precision, recall, accuracy, f1 = getMetrics(preds, test_y)
                        strlog = "{},{},{},{},{},{},{},{},{},{}".format(mode,target,mltype,prime_inclusion_str,product_id,version,precision,recall,accuracy,f1)
                        printLog(strlog,log_file)
                if mltype == 'demographics':
                    X,Y,test_x,test_y = LoadData_product(product_id,category,target,mltype,prime_inclusion)
                    product_intent_demographics_PrimeNotIncluded_model, product_intent_demographics_PrimeNotIncluded_graph = DM_product_intent_demographics_PrimeNotIncluded(run_id,tf.Graph())
                    with product_intent_demographics_PrimeNotIncluded_graph.as_default():
                        product_intent_demographics_PrimeNotIncluded_model.fit(
                            {'{}_input'.format(run_id): X},
                            {'{}_targets'.format(run_id): Y},
                            n_epoch=5,
                            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                            show_metric=True,
                            batch_size=None,
                            shuffle=False,
                            snapshot_step=None,
                            run_id=run_id
                            )
                        product_intent_demographics_PrimeNotIncluded_model.save(model_path)
                        preds = product_intent_demographics_PrimeNotIncluded_model.predict(test_x)
                        precision, recall, accuracy, f1 = getMetrics(preds, test_y)
                        strlog = "{},{},{},{},{},{},{},{},{},{}".format(mode,target,mltype,prime_inclusion_str,product_id,version,precision,recall,accuracy,f1)
                        printLog(strlog,log_file)
                if mltype == 'cm_demo':
                    X,Y,test_x,test_y = LoadData_product(product_id,category,target,mltype,prime_inclusion)
                    product_intent_cm_demo_PrimeNotIncluded_model, product_intent_cm_demo_PrimeNotIncluded_graph = DM_product_intent_cm_demo_PrimeNotIncluded(run_id,tf.Graph())
                    with product_intent_cm_demo_PrimeNotIncluded_graph.as_default():
                        product_intent_cm_demo_PrimeNotIncluded_model.fit(
                            {'{}_input'.format(run_id): X},
                            {'{}_targets'.format(run_id): Y},
                            n_epoch=5,
                            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                            show_metric=True,
                            batch_size=None,
                            shuffle=False,
                            snapshot_step=None,
                            run_id=run_id
                            )
                        product_intent_cm_demo_PrimeNotIncluded_model.save(model_path)
                        preds = product_intent_cm_demo_PrimeNotIncluded_model.predict(test_x)
                        precision, recall, accuracy, f1 = getMetrics(preds, test_y)
                        strlog = "{},{},{},{},{},{},{},{},{},{}".format(mode,target,mltype,prime_inclusion_str,product_id,version,precision,recall,accuracy,f1)
                        printLog(strlog,log_file)
            if prime_inclusion == True:
                if mltype == 'cm_only':
                    X,Y,test_x,test_y = LoadData_product(product_id,category,target,mltype,prime_inclusion)
                    product_intent_cm_only_PrimeIncluded_model, product_intent_cm_only_PrimeIncluded_graph = DM_product_intent_cm_only_PrimeIncluded(run_id,tf.Graph())
                    with product_intent_cm_only_PrimeIncluded_graph.as_default():
                        product_intent_cm_only_PrimeIncluded_model.fit(
                            {'{}_input'.format(run_id): X},
                            {'{}_targets'.format(run_id): Y},
                            n_epoch=5,
                            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                            show_metric=True,
                            batch_size=None,
                            shuffle=False,
                            snapshot_step=None,
                            run_id=run_id
                            )
                        product_intent_cm_only_PrimeIncluded_model.save(model_path)
                        preds = product_intent_cm_only_PrimeIncluded_model.predict(test_x)
                        precision, recall, accuracy, f1 = getMetrics(preds, test_y)
                        strlog = "{},{},{},{},{},{},{},{},{},{}".format(mode,target,mltype,prime_inclusion_str,product_id,version,precision,recall,accuracy,f1)
                        printLog(strlog,log_file)
                if mltype == 'demographics':
                    X,Y,test_x,test_y = LoadData_product(product_id,category,target,mltype,prime_inclusion)
                    product_intent_demographics_PrimeIncluded_model, product_intent_demographics_PrimeIncluded_graph = DM_product_intent_demographics_PrimeIncluded(run_id,tf.Graph())
                    with product_intent_demographics_PrimeIncluded_graph.as_default():
                        product_intent_demographics_PrimeIncluded_model.fit(
                            {'{}_input'.format(run_id): X},
                            {'{}_targets'.format(run_id): Y},
                            n_epoch=5,
                            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                            show_metric=True,
                            batch_size=None,
                            shuffle=False,
                            snapshot_step=None,
                            run_id=run_id
                            )
                        product_intent_demographics_PrimeIncluded_model.save(model_path)
                        preds = product_intent_demographics_PrimeIncluded_model.predict(test_x)
                        precision, recall, accuracy, f1 = getMetrics(preds, test_y)
                        strlog = "{},{},{},{},{},{},{},{},{},{}".format(mode,target,mltype,prime_inclusion_str,product_id,version,precision,recall,accuracy,f1)
                        printLog(strlog,log_file)
                if mltype == 'cm_demo':
                    X,Y,test_x,test_y = LoadData_product(product_id,category,target,mltype,prime_inclusion)
                    product_intent_cm_demo_PrimeIncluded_model, product_intent_cm_demo_PrimeIncluded_graph = DM_product_intent_cm_demo_PrimeIncluded(run_id,tf.Graph())
                    with product_intent_cm_demo_PrimeIncluded_graph.as_default():
                        product_intent_cm_demo_PrimeIncluded_model.fit(
                            {'{}_input'.format(run_id): X},
                            {'{}_targets'.format(run_id): Y},
                            n_epoch=5,
                            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                            show_metric=True,
                            batch_size=None,
                            shuffle=False,
                            snapshot_step=None,
                            run_id=run_id
                            )
                        product_intent_cm_demo_PrimeIncluded_model.save(model_path)
                        preds = product_intent_cm_demo_PrimeIncluded_model.predict(test_x)
                        precision, recall, accuracy, f1 = getMetrics(preds, test_y)
                        strlog = "{},{},{},{},{},{},{},{},{},{}".format(mode,target,mltype,prime_inclusion_str,product_id,version,precision,recall,accuracy,f1)
                        printLog(strlog,log_file)
    ######################
    if mode == 'user':
        if target == 'purchase':
            if prime_inclusion == False:
                if mltype == 'cm_only':
                    X,Y,test_x,test_y = LoadData_user(user_id,category,target,mltype,prime_inclusion)
                    user_purchase_cm_only_PrimeNotIncluded_model, user_purchase_cm_only_PrimeNotIncluded_graph = DM_user_purchase_cm_only_PrimeNotIncluded(run_id,tf.Graph())
                    with user_purchase_cm_only_PrimeNotIncluded_graph.as_default():
                        user_purchase_cm_only_PrimeNotIncluded_model.fit(
                            {'{}_input'.format(run_id): X},
                            {'{}_targets'.format(run_id): Y},
                            n_epoch=5,
                            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                            show_metric=True,
                            batch_size=None,
                            shuffle=False,
                            snapshot_step=None,
                            run_id=run_id
                            )
                        user_purchase_cm_only_PrimeNotIncluded_model.save(model_path)
                        preds = user_purchase_cm_only_PrimeNotIncluded_model.predict(test_x)
                        precision, recall, accuracy, f1 = getMetrics(preds, test_y)
                        strlog = "{},{},{},{},{},{},{},{},{},{}".format(mode,target,mltype,prime_inclusion_str,product_id,version,precision,recall,accuracy,f1)
                        printLog(strlog,log_file)
                if mltype == 'demographics':
                    X,Y,test_x,test_y = LoadData_user(user_id,category,target,mltype,prime_inclusion)
                    user_purchase_demographics_PrimeNotIncluded_model, user_purchase_demographics_PrimeNotIncluded_graph = DM_user_purchase_demographics_PrimeNotIncluded(run_id,tf.Graph())
                    with user_purchase_demographics_PrimeNotIncluded_graph.as_default():
                        user_purchase_demographics_PrimeNotIncluded_model.fit(
                            {'{}_input'.format(run_id): X},
                            {'{}_targets'.format(run_id): Y},
                            n_epoch=5,
                            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                            show_metric=True,
                            batch_size=None,
                            shuffle=False,
                            snapshot_step=None,
                            run_id=run_id
                            )
                        user_purchase_demographics_PrimeNotIncluded_model.save(model_path)
                        preds = user_purchase_demographics_PrimeNotIncluded_model.predict(test_x)
                        precision, recall, accuracy, f1 = getMetrics(preds, test_y)
                        strlog = "{},{},{},{},{},{},{},{},{},{}".format(mode,target,mltype,prime_inclusion_str,product_id,version,precision,recall,accuracy,f1)
                        printLog(strlog,log_file)
                if mltype == 'cm_demo':
                    X,Y,test_x,test_y = LoadData_user(user_id,category,target,mltype,prime_inclusion)
                    user_purchase_cm_demo_PrimeNotIncluded_model, user_purchase_cm_demo_PrimeNotIncluded_graph = DM_user_purchase_cm_demo_PrimeNotIncluded(run_id,tf.Graph())
                    with user_purchase_cm_demo_PrimeNotIncluded_graph.as_default():
                        user_purchase_cm_demo_PrimeNotIncluded_model.fit(
                            {'{}_input'.format(run_id): X},
                            {'{}_targets'.format(run_id): Y},
                            n_epoch=5,
                            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                            show_metric=True,
                            batch_size=None,
                            shuffle=False,
                            snapshot_step=None,
                            run_id=run_id
                            )
                        user_purchase_cm_demo_PrimeNotIncluded_model.save(model_path)
                        preds = user_purchase_cm_demo_PrimeNotIncluded_model.predict(test_x)
                        precision, recall, accuracy, f1 = getMetrics(preds, test_y)
                        strlog = "{},{},{},{},{},{},{},{},{},{}".format(mode,target,mltype,prime_inclusion_str,product_id,version,precision,recall,accuracy,f1)
                        printLog(strlog,log_file)
            if prime_inclusion == True:
                if mltype == 'cm_only':
                    X,Y,test_x,test_y = LoadData_user(user_id,category,target,mltype,prime_inclusion)
                    user_purchase_cm_only_PrimeIncluded_model, user_purchase_cm_only_PrimeIncluded_graph = DM_user_purchase_cm_only_PrimeIncluded(run_id,tf.Graph())
                    with user_purchase_cm_only_PrimeIncluded_graph.as_default():
                        user_purchase_cm_only_PrimeIncluded_model.fit(
                            {'{}_input'.format(run_id): X},
                            {'{}_targets'.format(run_id): Y},
                            n_epoch=5,
                            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                            show_metric=True,
                            batch_size=None,
                            shuffle=False,
                            snapshot_step=None,
                            run_id=run_id
                            )
                        user_purchase_cm_only_PrimeIncluded_model.save(model_path)
                        preds = user_purchase_cm_only_PrimeIncluded_model.predict(test_x)
                        precision, recall, accuracy, f1 = getMetrics(preds, test_y)
                        strlog = "{},{},{},{},{},{},{},{},{},{}".format(mode,target,mltype,prime_inclusion_str,product_id,version,precision,recall,accuracy,f1)
                        printLog(strlog,log_file)
                if mltype == 'demographics':
                    X,Y,test_x,test_y = LoadData_user(user_id,category,target,mltype,prime_inclusion)
                    user_purchase_demographics_PrimeIncluded_model, user_purchase_demographics_PrimeIncluded_graph = DM_user_purchase_demographics_PrimeIncluded(run_id,tf.Graph())
                    with user_purchase_demographics_PrimeIncluded_graph.as_default():
                        user_purchase_demographics_PrimeIncluded_model.fit(
                            {'{}_input'.format(run_id): X},
                            {'{}_targets'.format(run_id): Y},
                            n_epoch=5,
                            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                            show_metric=True,
                            batch_size=None,
                            shuffle=False,
                            snapshot_step=None,
                            run_id=run_id
                            )
                        user_purchase_demographics_PrimeIncluded_model.save(model_path)
                        preds = user_purchase_demographics_PrimeIncluded_model.predict(test_x)
                        precision, recall, accuracy, f1 = getMetrics(preds, test_y)
                        strlog = "{},{},{},{},{},{},{},{},{},{}".format(mode,target,mltype,prime_inclusion_str,product_id,version,precision,recall,accuracy,f1)
                        printLog(strlog,log_file)
                if mltype == 'cm_demo':
                    X,Y,test_x,test_y = LoadData_user(user_id,category,target,mltype,prime_inclusion)
                    user_purchase_cm_demo_PrimeIncluded_model, user_purchase_cm_demo_PrimeIncluded_graph = DM_user_purchase_cm_demo_PrimeIncluded(run_id,tf.Graph())
                    with user_purchase_cm_demo_PrimeIncluded_graph.as_default():
                        user_purchase_cm_demo_PrimeIncluded_model.fit(
                            {'{}_input'.format(run_id): X},
                            {'{}_targets'.format(run_id): Y},
                            n_epoch=5,
                            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                            show_metric=True,
                            batch_size=None,
                            shuffle=False,
                            snapshot_step=None,
                            run_id=run_id
                            )
                        user_purchase_cm_demo_PrimeIncluded_model.save(model_path)
                        preds = user_purchase_cm_demo_PrimeIncluded_model.predict(test_x)
                        precision, recall, accuracy, f1 = getMetrics(preds, test_y)
                        strlog = "{},{},{},{},{},{},{},{},{},{}".format(mode,target,mltype,prime_inclusion_str,product_id,version,precision,recall,accuracy,f1)
                        printLog(strlog,log_file)
        if target == 'intent':
            if prime_inclusion == False:
                if mltype == 'cm_only':
                    X,Y,test_x,test_y = LoadData_user(user_id,category,target,mltype,prime_inclusion)
                    user_intent_cm_only_PrimeNotIncluded_model, user_intent_cm_only_PrimeNotIncluded_graph = DM_user_intent_cm_only_PrimeNotIncluded(run_id,tf.Graph())
                    with user_intent_cm_only_PrimeNotIncluded_graph.as_default():
                        user_intent_cm_only_PrimeNotIncluded_model.fit(
                            {'{}_input'.format(run_id): X},
                            {'{}_targets'.format(run_id): Y},
                            n_epoch=5,
                            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                            show_metric=True,
                            batch_size=None,
                            shuffle=False,
                            snapshot_step=None,
                            run_id=run_id
                            )
                        user_intent_cm_only_PrimeNotIncluded_model.save(model_path)
                        preds = user_intent_cm_only_PrimeNotIncluded_model.predict(test_x)
                        precision, recall, accuracy, f1 = getMetrics(preds, test_y)
                        strlog = "{},{},{},{},{},{},{},{},{},{}".format(mode,target,mltype,prime_inclusion_str,product_id,version,precision,recall,accuracy,f1)
                        printLog(strlog,log_file)
                if mltype == 'demographics':
                    X,Y,test_x,test_y = LoadData_user(user_id,category,target,mltype,prime_inclusion)
                    user_intent_demographics_PrimeNotIncluded_model, user_intent_demographics_PrimeNotIncluded_graph = DM_user_intent_demographics_PrimeNotIncluded(run_id,tf.Graph())
                    with user_intent_demographics_PrimeNotIncluded_graph.as_default():
                        user_intent_demographics_PrimeNotIncluded_model.fit(
                            {'{}_input'.format(run_id): X},
                            {'{}_targets'.format(run_id): Y},
                            n_epoch=5,
                            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                            show_metric=True,
                            batch_size=None,
                            shuffle=False,
                            snapshot_step=None,
                            run_id=run_id
                            )
                        user_intent_demographics_PrimeNotIncluded_model.save(model_path)
                        preds = user_intent_demographics_PrimeNotIncluded_model.predict(test_x)
                        precision, recall, accuracy, f1 = getMetrics(preds, test_y)
                        strlog = "{},{},{},{},{},{},{},{},{},{}".format(mode,target,mltype,prime_inclusion_str,product_id,version,precision,recall,accuracy,f1)
                        printLog(strlog,log_file)
                if mltype == 'cm_demo':
                    X,Y,test_x,test_y = LoadData_user(user_id,category,target,mltype,prime_inclusion)
                    user_intent_cm_demo_PrimeNotIncluded_model, user_intent_cm_demo_PrimeNotIncluded_graph = DM_user_intent_cm_demo_PrimeNotIncluded(run_id,tf.Graph())
                    with user_intent_cm_demo_PrimeNotIncluded_graph.as_default():
                        user_intent_cm_demo_PrimeNotIncluded_model.fit(
                            {'{}_input'.format(run_id): X},
                            {'{}_targets'.format(run_id): Y},
                            n_epoch=5,
                            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                            show_metric=True,
                            batch_size=None,
                            shuffle=False,
                            snapshot_step=None,
                            run_id=run_id
                            )
                        user_intent_cm_demo_PrimeNotIncluded_model.save(model_path)
                        preds = user_intent_cm_demo_PrimeNotIncluded_model.predict(test_x)
                        precision, recall, accuracy, f1 = getMetrics(preds, test_y)
                        strlog = "{},{},{},{},{},{},{},{},{},{}".format(mode,target,mltype,prime_inclusion_str,product_id,version,precision,recall,accuracy,f1)
                        printLog(strlog,log_file)
            if prime_inclusion == True:
                if mltype == 'cm_only':
                    X,Y,test_x,test_y = LoadData_user(user_id,category,target,mltype,prime_inclusion)
                    user_intent_cm_only_PrimeIncluded_model, user_intent_cm_only_PrimeIncluded_graph = DM_user_intent_cm_only_PrimeIncluded(run_id,tf.Graph())
                    with user_intent_cm_only_PrimeIncluded_graph.as_default():
                        user_intent_cm_only_PrimeIncluded_model.fit(
                            {'{}_input'.format(run_id): X},
                            {'{}_targets'.format(run_id): Y},
                            n_epoch=5,
                            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                            show_metric=True,
                            batch_size=None,
                            shuffle=False,
                            snapshot_step=None,
                            run_id=run_id
                            )
                        user_intent_cm_only_PrimeIncluded_model.save(model_path)
                        preds = user_intent_cm_only_PrimeIncluded_model.predict(test_x)
                        precision, recall, accuracy, f1 = getMetrics(preds, test_y)
                        strlog = "{},{},{},{},{},{},{},{},{},{}".format(mode,target,mltype,prime_inclusion_str,product_id,version,precision,recall,accuracy,f1)
                        printLog(strlog,log_file)
                if mltype == 'demographics':
                    X,Y,test_x,test_y = LoadData_user(user_id,category,target,mltype,prime_inclusion)
                    user_intent_demographics_PrimeIncluded_model, user_intent_demographics_PrimeIncluded_graph = DM_user_intent_demographics_PrimeIncluded(run_id,tf.Graph())
                    with user_intent_demographics_PrimeIncluded_graph.as_default():
                        user_intent_demographics_PrimeIncluded_model.fit(
                            {'{}_input'.format(run_id): X},
                            {'{}_targets'.format(run_id): Y},
                            n_epoch=5,
                            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                            show_metric=True,
                            batch_size=None,
                            shuffle=False,
                            snapshot_step=None,
                            run_id=run_id
                            )
                        user_intent_demographics_PrimeIncluded_model.save(model_path)
                        preds = user_intent_demographics_PrimeIncluded_model.predict(test_x)
                        precision, recall, accuracy, f1 = getMetrics(preds, test_y)
                        strlog = "{},{},{},{},{},{},{},{},{},{}".format(mode,target,mltype,prime_inclusion_str,product_id,version,precision,recall,accuracy,f1)
                        printLog(strlog,log_file)
                if mltype == 'cm_demo':
                    X,Y,test_x,test_y = LoadData_user(user_id,category,target,mltype,prime_inclusion)
                    user_intent_cm_demo_PrimeIncluded_model, user_intent_cm_demo_PrimeIncluded_graph = DM_user_intent_cm_demo_PrimeIncluded(run_id,tf.Graph())
                    with user_intent_cm_demo_PrimeIncluded_graph.as_default():
                        user_intent_cm_demo_PrimeIncluded_model.fit(
                            {'{}_input'.format(run_id): X},
                            {'{}_targets'.format(run_id): Y},
                            n_epoch=5,
                            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
                            show_metric=True,
                            batch_size=None,
                            shuffle=False,
                            snapshot_step=None,
                            run_id=run_id
                            )
                        user_intent_cm_demo_PrimeIncluded_model.save(model_path)
                        preds = user_intent_cm_demo_PrimeIncluded_model.predict(test_x)
                        precision, recall, accuracy, f1 = getMetrics(preds, test_y)
                        strlog = "{},{},{},{},{},{},{},{},{},{}".format(mode,target,mltype,prime_inclusion_str,product_id,version,precision,recall,accuracy,f1)
                        printLog(strlog,log_file)
        ######################
    print_log_instructions()

###################
###################
###################
###################
###################
###################

def main():
    targets = ['purchase','intent']
    categories = [2,4,5,3,0,1]
    prime_inclusions = [False, True]
    types = ['cm_only','demographics','cm_demo']
    ########
    mode='product'
    log_file = MakeLogFile('NeuralNetworkTests_product.csv',server=True)
    titles = ['Base', 'Target', 'Model','PrimeInclusion', 'Product_ID','version','Precision','Recall','Accuracy','F1']
    strlog = ','.join(titles)
    printLog(strlog,log_file)
    ###
    products = getProductList()
    for target in targets:
        for category in categories:
            for prime_inclusion in prime_inclusions:
                for mltype in types:
                    for product_id in products:
                        tf.reset_default_graph()
                        base_id = product_id
                        Train(base_id, category, mode, target, mltype, prime_inclusion)
    #############
    mode = 'user'
    log_file = MakeLogFile('NeuralNetworkTests_user.csv',server=True)
    titles = ['Base', 'Target', 'Model','PrimeInclusion', 'User_ID','version','Precision','Recall','Accuracy','F1']
    strlog = ','.join(titles)
    printLog(strlog,log_file)
    ###
    _,main_data = readCSV(getMainDataCSVPath(server=server))
    users = [user[0] for user in main_data]
    for target in targets:
        for category in categories:
            for prime_inclusion in prime_inclusions:
                for mltype in types:
                    for user_id in users:
                        tf.reset_default_graph()
                        base_id = user_id
                        Train(base_id, category, mode, target, mltype, prime_inclusion)
    

if __name__ == '__main__':
    main()

