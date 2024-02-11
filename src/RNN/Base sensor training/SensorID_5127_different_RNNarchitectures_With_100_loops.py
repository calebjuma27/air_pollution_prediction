#-------------------------------------------------------------------------------
# Name:        SensorID5127
# Purpose:     Finding best RNNcombination
#
# Created:     12-12-2019
# Copyright:   (c) t-82maca1mpg 2019
##special tahnks to Weimin Wang (https://weiminwang.blog/2017/09/29/multivariate-time-series-forecast-using-seq2seq-in-tensorflow/)
##Modified by   Caleb Juma

#-------------------------------------------------------------------------------

# Imports
from statistics import mean

# DL libraries
import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes

# Data Manipulation
import numpy as np
import pandas as pd
import random
import math

# Files/OS
import os
import copy

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Benchmarking
import time

# Error Analysis
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


##################################################################################################################
##THe if statement and the vscor and tscor are used a way of measuring the accuracy of each...
#...RNN variant with respect to the final Variance score and time taken.
#They can be removed without affecting the process.

#variance score for each of the 6 RNN variants being tested
vscor1=[]
vscor2=[]
vscor3=[]
vscor4=[]
vscor5=[]
vscor6=[]

#time taken for each of the 6 RNN variants being trained
trscor1=[]
trscor2=[]
trscor3=[]
trscor4=[]
trscor5=[]
trscor6=[]

#time taken for each of the 6 RNN variants being tested
tescor1=[]
tescor2=[]
tescor3=[]
tescor4=[]
tescor5=[]
tescor6=[]

#RMSE for each of the 6 RNN variants being tested
ramse1=[]
ramse2=[]
ramse3=[]
ramse4=[]
ramse5=[]
ramse6=[]


for i in range(100):

    #Prepare input data

    df = pd.read_csv('C:/Users/t-82maca1mpg/Desktop/MSCPG_CalebJuma/RNN/data in csv/sensorID_5127_csv_table_sorted.csv')#put in a diff folder to the py file
    print(df.head())

    #visualize the dataset
    cols_to_plot = ["p1hr","p2hr", "temperatur", "humidity", "pressure", "windspeed", "winddirect"]
    i = 1
    # plot each column
    ##plt.figure(figsize = (10,12))
    ##for col in cols_to_plot:
    ##    plt.subplot(len(cols_to_plot), 1, i)
    ##    plt.plot(df[col])
    ##    plt.title(col, y=0.5, loc='left')
    ##    i += 1
    ##plt.show()

    ## Split into train and test - Ithe month of Nov was used as the test datasource
    df_train = df.iloc[:(-256), :].copy()# -256 refers to counting 256 from the last record. Thus from first record to (end-256)
    df_test = df.iloc[-256:, :].copy()


    ## take out the useful columns for modeling - you may also keep 'hour', 'day' or 'month' and to see if that will improve your accuracy
    X_train = df_train.loc[:, ["p1hr", "temperatur", "humidity", "pressure", "windspeed", "winddirect","Month", "Date", "gmt"]].values.copy()
    X_test = df_test.loc[:, ["p1hr", "temperatur", "humidity", "pressure", "windspeed", "winddirect","Month", "Date", "gmt"]].values.copy()
    y_train = df_train['p2hr'].values.copy().reshape(-1, 1)
    y_test = df_test['p2hr'].values.copy().reshape(-1, 1)


    ## z-score transform x -to ensure the averages are computed from the same base
    for i in range(X_train.shape[1]):
        temp_mean = X_train[:, i].mean()
        temp_std = X_train[:, i].std()
        X_train[:, i] = (X_train[:, i] - temp_mean) / temp_std
        X_test[:, i] = (X_test[:, i] - temp_mean) / temp_std

    ## z-score transform y
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    input_seq_len = 20
    output_seq_len = 1


    #transform into 3D format for timeseries ie having batch size, time_steps, and feature dimension
    #these fuctions will later be used to generate tarining samples and test samples
    def generate_train_samples(x = X_train, y = y_train, batch_size = 24, input_seq_len = input_seq_len, output_seq_len = output_seq_len):

        total_start_points = len(x) - input_seq_len - output_seq_len
        print("2nd lenx=", len(x))
        print ("total start point =",total_start_points)
        start_x_idx = np.random.choice(range(total_start_points), batch_size, replace = False)

        input_batch_idxs = [list(range(i, i+input_seq_len)) for i in start_x_idx]
        input_seq = np.take(x, input_batch_idxs, axis = 0)

        output_batch_idxs = [list(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in start_x_idx]
        output_seq = np.take(y, output_batch_idxs, axis = 0)

        return input_seq, output_seq # in shape: (batch_size, time_steps, feature_dim)



    def generate_test_samples(x = X_test, y = y_test, input_seq_len = input_seq_len, output_seq_len = output_seq_len):

        total_samples = x.shape[0]

        input_batch_idxs = [list(range(i, i+input_seq_len)) for i in range((total_samples-input_seq_len-output_seq_len))]
        input_seq = np.take(x, input_batch_idxs, axis = 0)

        output_batch_idxs = [list(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in range((total_samples-input_seq_len-output_seq_len))]
        output_seq = np.take(y, output_batch_idxs, axis = 0)

        return input_seq, output_seq



    #########################################################################################################################

    #Build the Graphs or models
    ## Parameters
    learning_rate = 0.01
    lambda_l2_reg = 0.003

    ## Network Parameters
    # length of input signals
    input_seq_len = input_seq_len
    # length of output signals
    output_seq_len = output_seq_len
    # size of LSTM Cell
    hidden_dim =14
    # num of input signals
    input_dim = X_train.shape[1]
    # num of output signals
    output_dim = y_train.shape[1]
    # num of stacked lstm layers
    num_stacked_layers = 2
    # gradient clipping - to avoid gradient exploding
    GRADIENT_CLIPPING = 1.5


    ### A) USING ADAM OPTIMIZER

    ###RNN Architecture 1-LSTM (std) with Adam optimizer

    def build_graph(feed_previous = False):

        tf.reset_default_graph()

        global_step = tf.Variable(
                      initial_value=0,
                      name="global_step",
                      trainable=False,
                      collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        weights = {
            'out': tf.get_variable('Weights_out', \
                                   shape = [hidden_dim, output_dim], \
                                   dtype = tf.float32, \
                                   initializer = tf.truncated_normal_initializer()),
        }
        biases = {
            'out': tf.get_variable('Biases_out', \
                                   shape = [output_dim], \
                                   dtype = tf.float32, \
                                   initializer = tf.constant_initializer(0.)),
        }

        with tf.variable_scope('Seq2seq'):
            # Encoder: inputs
            enc_inp = [
                tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
                   for t in range(input_seq_len)
            ]

            # Decoder: target outputs
            target_seq = [
                tf.placeholder(tf.float32, shape=(None, output_dim), name="y".format(t))
                  for t in range(output_seq_len)
            ]

            # Give a "GO" token to the decoder.
            # If dec_inp are fed into decoder as inputs, this is 'guided' training; otherwise only the
            # first element will be fed as decoder input which is then 'un-guided'
            dec_inp = [ tf.zeros_like(target_seq[0], dtype=tf.float32, name="GO") ] + target_seq[:-1]

            with tf.variable_scope('LSTMCell'):
                cells = []
                for i in range(num_stacked_layers):
                    with tf.variable_scope('RNN_{}'.format(i)):
                        cells.append(tf.contrib.rnn.LSTMCell(hidden_dim))
                cell = tf.contrib.rnn.MultiRNNCell(cells)

            def _rnn_decoder(decoder_inputs,
                            initial_state,
                            cell,
                            loop_function=None,
                            scope=None):
              """RNN decoder for the sequence-to-sequence model.
              Args:
                decoder_inputs: A list of 2D Tensors [batch_size x input_size].
                initial_state: 2D Tensor with shape [batch_size x cell.state_size].
                cell: rnn_cell.RNNCell defining the cell function and size.
                loop_function: If not None, this function will be applied to the i-th output
                  in order to generate the i+1-st input, and decoder_inputs will be ignored,
                  except for the first element ("GO" symbol). This can be used for decoding,
                  but also for training to emulate http://arxiv.org/abs/1506.03099.
                  Signature -- loop_function(prev, i) = next
                    * prev is a 2D Tensor of shape [batch_size x output_size],
                    * i is an integer, the step number (when advanced control is needed),
                    * next is a 2D Tensor of shape [batch_size x input_size].
                scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
              Returns:
                A tuple of the form (outputs, state), where:
                  outputs: A list of the same length as decoder_inputs of 2D Tensors with
                    shape [batch_size x output_size] containing generated outputs.
                  state: The state of each cell at the final time-step.
                    It is a 2D Tensor of shape [batch_size x cell.state_size].
                    (Note that in some cases, like basic RNN cell or GRU cell, outputs and
                     states can be the same. They are different for LSTM cells though.)
              """
              with variable_scope.variable_scope(scope or "rnn_decoder"):
                state = initial_state
                outputs = []
                prev = None
                for i, inp in enumerate(decoder_inputs):
                  if loop_function is not None and prev is not None:
                    with variable_scope.variable_scope("loop_function", reuse=True):
                      inp = loop_function(prev, i)
                  if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()
                  output, state = cell(inp, state)
                  outputs.append(output)
                  if loop_function is not None:
                    prev = output
              return outputs, state

            def _basic_rnn_seq2seq(encoder_inputs,
                                  decoder_inputs,
                                  cell,
                                  feed_previous,
                                  dtype=dtypes.float32,
                                  scope=None):
              """Basic RNN sequence-to-sequence model.
              This model first runs an RNN to encode encoder_inputs into a state vector,
              then runs decoder, initialized with the last encoder state, on decoder_inputs.
              Encoder and decoder use the same RNN cell type, but don't share parameters.
              Args:
                encoder_inputs: A list of 2D Tensors [batch_size x input_size].
                decoder_inputs: A list of 2D Tensors [batch_size x input_size].
                feed_previous: Boolean; if True, only the first of decoder_inputs will be
                  used (the "GO" symbol), all other inputs will be generated by the previous
                  decoder output using _loop_function below. If False, decoder_inputs are used
                  as given (the standard decoder case).
                dtype: The dtype of the initial state of the RNN cell (default: tf.float32).
                scope: VariableScope for the created subgraph; default: "basic_rnn_seq2seq".
              Returns:
                A tuple of the form (outputs, state), where:
                  outputs: A list of the same length as decoder_inputs of 2D Tensors with
                    shape [batch_size x output_size] containing the generated outputs.
                  state: The state of each decoder cell in the final time-step.
                    It is a 2D Tensor of shape [batch_size x cell.state_size].
              """
              with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
                enc_cell = copy.deepcopy(cell)
                _, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)
                if feed_previous:
                    return _rnn_decoder(decoder_inputs, enc_state, cell, _loop_function)
                else:
                    return _rnn_decoder(decoder_inputs, enc_state, cell)

            def _loop_function(prev, _):
              '''Naive implementation of loop function for _rnn_decoder. Transform prev from
              dimension [batch_size x hidden_dim] to [batch_size x output_dim], which will be
              used as decoder input of next time step '''
              return tf.matmul(prev, weights['out']) + biases['out']

            dec_outputs, dec_memory = _basic_rnn_seq2seq(
                enc_inp,
                dec_inp,
                cell,
                feed_previous = feed_previous
            )

            reshaped_outputs = [tf.matmul(i, weights['out']) + biases['out'] for i in dec_outputs]

        # Training loss and optimizer
        with tf.variable_scope('Loss'):
            # L2 loss
            output_loss = 0
            for _y, _Y in zip(reshaped_outputs, target_seq):
                output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))

            # L2 regularization for weights and biases
            reg_loss = 0
            for tf_var in tf.trainable_variables():
                if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                    reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

            loss = output_loss + lambda_l2_reg * reg_loss

        with tf.variable_scope('Optimizer'):
            optimizer = tf.contrib.layers.optimize_loss(
                    loss=loss,
                    learning_rate=learning_rate,
                    global_step=global_step,
                    optimizer='Adam',
                    clip_gradients=GRADIENT_CLIPPING)

        saver = tf.train.Saver

        return dict(
            enc_inp = enc_inp,
            target_seq = target_seq,
            train_op = optimizer,
            loss=loss,
            saver = saver,
            reshaped_outputs = reshaped_outputs,
            )







    ###RNN Architecture 2-LSTM peepholes with Adam optimizer
    def build_graph2(feed_previous = False):

        tf.reset_default_graph()

        global_step = tf.Variable(
                      initial_value=0,
                      name="global_step",
                      trainable=False,
                      collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        weights = {
            'out': tf.get_variable('Weights_out', \
                                   shape = [hidden_dim, output_dim], \
                                   dtype = tf.float32, \
                                   initializer = tf.truncated_normal_initializer()),
        }
        biases = {
            'out': tf.get_variable('Biases_out', \
                                   shape = [output_dim], \
                                   dtype = tf.float32, \
                                   initializer = tf.constant_initializer(0.)),
        }

        with tf.variable_scope('Seq2seq'):
            # Encoder: inputs
            enc_inp = [
                tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
                   for t in range(input_seq_len)
            ]

            # Decoder: target outputs
            target_seq = [
                tf.placeholder(tf.float32, shape=(None, output_dim), name="y".format(t))
                  for t in range(output_seq_len)
            ]

            # Give a "GO" token to the decoder.
            # If dec_inp are fed into decoder as inputs, this is 'guided' training; otherwise only the
            # first element will be fed as decoder input which is then 'un-guided'
            dec_inp = [ tf.zeros_like(target_seq[0], dtype=tf.float32, name="GO") ] + target_seq[:-1]

            with tf.variable_scope('LSTMCell'):
                cells = []
                for i in range(num_stacked_layers):
                    with tf.variable_scope('RNN_{}'.format(i)):
                        cells.append(tf.contrib.rnn.LSTMCell(hidden_dim, use_peepholes= True))
                cell = tf.contrib.rnn.MultiRNNCell(cells)

            def _rnn_decoder(decoder_inputs,
                            initial_state,
                            cell,
                            loop_function=None,
                            scope=None):
              """RNN decoder for the sequence-to-sequence model.
              Args:
                decoder_inputs: A list of 2D Tensors [batch_size x input_size].
                initial_state: 2D Tensor with shape [batch_size x cell.state_size].
                cell: rnn_cell.RNNCell defining the cell function and size.
                loop_function: If not None, this function will be applied to the i-th output
                  in order to generate the i+1-st input, and decoder_inputs will be ignored,
                  except for the first element ("GO" symbol). This can be used for decoding,
                  but also for training to emulate http://arxiv.org/abs/1506.03099.
                  Signature -- loop_function(prev, i) = next
                    * prev is a 2D Tensor of shape [batch_size x output_size],
                    * i is an integer, the step number (when advanced control is needed),
                    * next is a 2D Tensor of shape [batch_size x input_size].
                scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
              Returns:
                A tuple of the form (outputs, state), where:
                  outputs: A list of the same length as decoder_inputs of 2D Tensors with
                    shape [batch_size x output_size] containing generated outputs.
                  state: The state of each cell at the final time-step.
                    It is a 2D Tensor of shape [batch_size x cell.state_size].
                    (Note that in some cases, like basic RNN cell or GRU cell, outputs and
                     states can be the same. They are different for LSTM cells though.)
              """
              with variable_scope.variable_scope(scope or "rnn_decoder"):
                state = initial_state
                outputs = []
                prev = None
                for i, inp in enumerate(decoder_inputs):
                  if loop_function is not None and prev is not None:
                    with variable_scope.variable_scope("loop_function", reuse=True):
                      inp = loop_function(prev, i)
                  if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()
                  output, state = cell(inp, state)
                  outputs.append(output)
                  if loop_function is not None:
                    prev = output
              return outputs, state

            def _basic_rnn_seq2seq(encoder_inputs,
                                  decoder_inputs,
                                  cell,
                                  feed_previous,
                                  dtype=dtypes.float32,
                                  scope=None):
              """Basic RNN sequence-to-sequence model.
              This model first runs an RNN to encode encoder_inputs into a state vector,
              then runs decoder, initialized with the last encoder state, on decoder_inputs.
              Encoder and decoder use the same RNN cell type, but don't share parameters.
              Args:
                encoder_inputs: A list of 2D Tensors [batch_size x input_size].
                decoder_inputs: A list of 2D Tensors [batch_size x input_size].
                feed_previous: Boolean; if True, only the first of decoder_inputs will be
                  used (the "GO" symbol), all other inputs will be generated by the previous
                  decoder output using _loop_function below. If False, decoder_inputs are used
                  as given (the standard decoder case).
                dtype: The dtype of the initial state of the RNN cell (default: tf.float32).
                scope: VariableScope for the created subgraph; default: "basic_rnn_seq2seq".
              Returns:
                A tuple of the form (outputs, state), where:
                  outputs: A list of the same length as decoder_inputs of 2D Tensors with
                    shape [batch_size x output_size] containing the generated outputs.
                  state: The state of each decoder cell in the final time-step.
                    It is a 2D Tensor of shape [batch_size x cell.state_size].
              """
              with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
                enc_cell = copy.deepcopy(cell)
                _, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)
                if feed_previous:
                    return _rnn_decoder(decoder_inputs, enc_state, cell, _loop_function)
                else:
                    return _rnn_decoder(decoder_inputs, enc_state, cell)

            def _loop_function(prev, _):
              '''Naive implementation of loop function for _rnn_decoder. Transform prev from
              dimension [batch_size x hidden_dim] to [batch_size x output_dim], which will be
              used as decoder input of next time step '''
              return tf.matmul(prev, weights['out']) + biases['out']

            dec_outputs, dec_memory = _basic_rnn_seq2seq(
                enc_inp,
                dec_inp,
                cell,
                feed_previous = feed_previous
            )

            reshaped_outputs = [tf.matmul(i, weights['out']) + biases['out'] for i in dec_outputs]

        # Training loss and optimizer
        with tf.variable_scope('Loss'):
            # L2 loss
            output_loss = 0
            for _y, _Y in zip(reshaped_outputs, target_seq):
                output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))

            # L2 regularization for weights and biases
            reg_loss = 0
            for tf_var in tf.trainable_variables():
                if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                    reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

            loss = output_loss + lambda_l2_reg * reg_loss

        with tf.variable_scope('Optimizer'):
            optimizer = tf.contrib.layers.optimize_loss(
                    loss=loss,
                    learning_rate=learning_rate,
                    global_step=global_step,
                    optimizer='Adam',
                    clip_gradients=GRADIENT_CLIPPING)

        saver = tf.train.Saver

        return dict(
            enc_inp = enc_inp,
            target_seq = target_seq,
            train_op = optimizer,
            loss=loss,
            saver = saver,
            reshaped_outputs = reshaped_outputs,
            )







    ###RNN Architecture 3-GRU with Adam optimizer
    def build_graph3(feed_previous = False):

        tf.reset_default_graph()

        global_step = tf.Variable(
                      initial_value=0,
                      name="global_step",
                      trainable=False,
                      collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        weights = {
            'out': tf.get_variable('Weights_out', \
                                   shape = [hidden_dim, output_dim], \
                                   dtype = tf.float32, \
                                   initializer = tf.truncated_normal_initializer()),
        }
        biases = {
            'out': tf.get_variable('Biases_out', \
                                   shape = [output_dim], \
                                   dtype = tf.float32, \
                                   initializer = tf.constant_initializer(0.)),
        }

        with tf.variable_scope('Seq2seq'):
            # Encoder: inputs
            enc_inp = [
                tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
                   for t in range(input_seq_len)
            ]

            # Decoder: target outputs
            target_seq = [
                tf.placeholder(tf.float32, shape=(None, output_dim), name="y".format(t))
                  for t in range(output_seq_len)
            ]

            # Give a "GO" token to the decoder.
            # If dec_inp are fed into decoder as inputs, this is 'guided' training; otherwise only the
            # first element will be fed as decoder input which is then 'un-guided'
            dec_inp = [ tf.zeros_like(target_seq[0], dtype=tf.float32, name="GO") ] + target_seq[:-1]

            with tf.variable_scope('GRUCell'):
                cells = []
                for i in range(num_stacked_layers):
                    with tf.variable_scope('RNN_{}'.format(i)):
                        cells.append(tf.contrib.rnn.GRUCell(hidden_dim))
                cell = tf.contrib.rnn.MultiRNNCell(cells)

            def _rnn_decoder(decoder_inputs,
                            initial_state,
                            cell,
                            loop_function=None,
                            scope=None):
              """RNN decoder for the sequence-to-sequence model.
              Args:
                decoder_inputs: A list of 2D Tensors [batch_size x input_size].
                initial_state: 2D Tensor with shape [batch_size x cell.state_size].
                cell: rnn_cell.RNNCell defining the cell function and size.
                loop_function: If not None, this function will be applied to the i-th output
                  in order to generate the i+1-st input, and decoder_inputs will be ignored,
                  except for the first element ("GO" symbol). This can be used for decoding,
                  but also for training to emulate http://arxiv.org/abs/1506.03099.
                  Signature -- loop_function(prev, i) = next
                    * prev is a 2D Tensor of shape [batch_size x output_size],
                    * i is an integer, the step number (when advanced control is needed),
                    * next is a 2D Tensor of shape [batch_size x input_size].
                scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
              Returns:
                A tuple of the form (outputs, state), where:
                  outputs: A list of the same length as decoder_inputs of 2D Tensors with
                    shape [batch_size x output_size] containing generated outputs.
                  state: The state of each cell at the final time-step.
                    It is a 2D Tensor of shape [batch_size x cell.state_size].
                    (Note that in some cases, like basic RNN cell or GRU cell, outputs and
                     states can be the same. They are different for LSTM cells though.)
              """
              with variable_scope.variable_scope(scope or "rnn_decoder"):
                state = initial_state
                outputs = []
                prev = None
                for i, inp in enumerate(decoder_inputs):
                  if loop_function is not None and prev is not None:
                    with variable_scope.variable_scope("loop_function", reuse=True):
                      inp = loop_function(prev, i)
                  if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()
                  output, state = cell(inp, state)
                  outputs.append(output)
                  if loop_function is not None:
                    prev = output
              return outputs, state

            def _basic_rnn_seq2seq(encoder_inputs,
                                  decoder_inputs,
                                  cell,
                                  feed_previous,
                                  dtype=dtypes.float32,
                                  scope=None):
              """Basic RNN sequence-to-sequence model.
              This model first runs an RNN to encode encoder_inputs into a state vector,
              then runs decoder, initialized with the last encoder state, on decoder_inputs.
              Encoder and decoder use the same RNN cell type, but don't share parameters.
              Args:
                encoder_inputs: A list of 2D Tensors [batch_size x input_size].
                decoder_inputs: A list of 2D Tensors [batch_size x input_size].
                feed_previous: Boolean; if True, only the first of decoder_inputs will be
                  used (the "GO" symbol), all other inputs will be generated by the previous
                  decoder output using _loop_function below. If False, decoder_inputs are used
                  as given (the standard decoder case).
                dtype: The dtype of the initial state of the RNN cell (default: tf.float32).
                scope: VariableScope for the created subgraph; default: "basic_rnn_seq2seq".
              Returns:
                A tuple of the form (outputs, state), where:
                  outputs: A list of the same length as decoder_inputs of 2D Tensors with
                    shape [batch_size x output_size] containing the generated outputs.
                  state: The state of each decoder cell in the final time-step.
                    It is a 2D Tensor of shape [batch_size x cell.state_size].
              """
              with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
                enc_cell = copy.deepcopy(cell)
                _, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)
                if feed_previous:
                    return _rnn_decoder(decoder_inputs, enc_state, cell, _loop_function)
                else:
                    return _rnn_decoder(decoder_inputs, enc_state, cell)

            def _loop_function(prev, _):
              '''Naive implementation of loop function for _rnn_decoder. Transform prev from
              dimension [batch_size x hidden_dim] to [batch_size x output_dim], which will be
              used as decoder input of next time step '''
              return tf.matmul(prev, weights['out']) + biases['out']

            dec_outputs, dec_memory = _basic_rnn_seq2seq(
                enc_inp,
                dec_inp,
                cell,
                feed_previous = feed_previous
            )

            reshaped_outputs = [tf.matmul(i, weights['out']) + biases['out'] for i in dec_outputs]

        # Training loss and optimizer
        with tf.variable_scope('Loss'):
            # L2 loss
            output_loss = 0
            for _y, _Y in zip(reshaped_outputs, target_seq):
                output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))

            # L2 regularization for weights and biases
            reg_loss = 0
            for tf_var in tf.trainable_variables():
                if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                    reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

            loss = output_loss + lambda_l2_reg * reg_loss

        with tf.variable_scope('Optimizer'):
            optimizer = tf.contrib.layers.optimize_loss(
                    loss=loss,
                    learning_rate=learning_rate,
                    global_step=global_step,
                    optimizer='Adam',
                    clip_gradients=GRADIENT_CLIPPING)

        saver = tf.train.Saver

        return dict(
            enc_inp = enc_inp,
            target_seq = target_seq,
            train_op = optimizer,
            loss=loss,
            saver = saver,
            reshaped_outputs = reshaped_outputs,
            )







    ###################################################################################################


    ### B) USING SGD(Stochastic Gradient Descent) OPTIMIZER

    ###RNN Architecture 4-LSTM (std) with SGD optimizer

    def build_graph4(feed_previous = False):

        tf.reset_default_graph()

        global_step = tf.Variable(
                      initial_value=0,
                      name="global_step",
                      trainable=False,
                      collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        weights = {
            'out': tf.get_variable('Weights_out', \
                                   shape = [hidden_dim, output_dim], \
                                   dtype = tf.float32, \
                                   initializer = tf.truncated_normal_initializer()),
        }
        biases = {
            'out': tf.get_variable('Biases_out', \
                                   shape = [output_dim], \
                                   dtype = tf.float32, \
                                   initializer = tf.constant_initializer(0.)),
        }

        with tf.variable_scope('Seq2seq'):
            # Encoder: inputs
            enc_inp = [
                tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
                   for t in range(input_seq_len)
            ]

            # Decoder: target outputs
            target_seq = [
                tf.placeholder(tf.float32, shape=(None, output_dim), name="y".format(t))
                  for t in range(output_seq_len)
            ]

            # Give a "GO" token to the decoder.
            # If dec_inp are fed into decoder as inputs, this is 'guided' training; otherwise only the
            # first element will be fed as decoder input which is then 'un-guided'
            dec_inp = [ tf.zeros_like(target_seq[0], dtype=tf.float32, name="GO") ] + target_seq[:-1]

            with tf.variable_scope('LSTMCell'):
                cells = []
                for i in range(num_stacked_layers):
                    with tf.variable_scope('RNN_{}'.format(i)):
                        cells.append(tf.contrib.rnn.LSTMCell(hidden_dim))
                cell = tf.contrib.rnn.MultiRNNCell(cells)

            def _rnn_decoder(decoder_inputs,
                            initial_state,
                            cell,
                            loop_function=None,
                            scope=None):
              """RNN decoder for the sequence-to-sequence model.
              Args:
                decoder_inputs: A list of 2D Tensors [batch_size x input_size].
                initial_state: 2D Tensor with shape [batch_size x cell.state_size].
                cell: rnn_cell.RNNCell defining the cell function and size.
                loop_function: If not None, this function will be applied to the i-th output
                  in order to generate the i+1-st input, and decoder_inputs will be ignored,
                  except for the first element ("GO" symbol). This can be used for decoding,
                  but also for training to emulate http://arxiv.org/abs/1506.03099.
                  Signature -- loop_function(prev, i) = next
                    * prev is a 2D Tensor of shape [batch_size x output_size],
                    * i is an integer, the step number (when advanced control is needed),
                    * next is a 2D Tensor of shape [batch_size x input_size].
                scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
              Returns:
                A tuple of the form (outputs, state), where:
                  outputs: A list of the same length as decoder_inputs of 2D Tensors with
                    shape [batch_size x output_size] containing generated outputs.
                  state: The state of each cell at the final time-step.
                    It is a 2D Tensor of shape [batch_size x cell.state_size].
                    (Note that in some cases, like basic RNN cell or GRU cell, outputs and
                     states can be the same. They are different for LSTM cells though.)
              """
              with variable_scope.variable_scope(scope or "rnn_decoder"):
                state = initial_state
                outputs = []
                prev = None
                for i, inp in enumerate(decoder_inputs):
                  if loop_function is not None and prev is not None:
                    with variable_scope.variable_scope("loop_function", reuse=True):
                      inp = loop_function(prev, i)
                  if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()
                  output, state = cell(inp, state)
                  outputs.append(output)
                  if loop_function is not None:
                    prev = output
              return outputs, state

            def _basic_rnn_seq2seq(encoder_inputs,
                                  decoder_inputs,
                                  cell,
                                  feed_previous,
                                  dtype=dtypes.float32,
                                  scope=None):
              """Basic RNN sequence-to-sequence model.
              This model first runs an RNN to encode encoder_inputs into a state vector,
              then runs decoder, initialized with the last encoder state, on decoder_inputs.
              Encoder and decoder use the same RNN cell type, but don't share parameters.
              Args:
                encoder_inputs: A list of 2D Tensors [batch_size x input_size].
                decoder_inputs: A list of 2D Tensors [batch_size x input_size].
                feed_previous: Boolean; if True, only the first of decoder_inputs will be
                  used (the "GO" symbol), all other inputs will be generated by the previous
                  decoder output using _loop_function below. If False, decoder_inputs are used
                  as given (the standard decoder case).
                dtype: The dtype of the initial state of the RNN cell (default: tf.float32).
                scope: VariableScope for the created subgraph; default: "basic_rnn_seq2seq".
              Returns:
                A tuple of the form (outputs, state), where:
                  outputs: A list of the same length as decoder_inputs of 2D Tensors with
                    shape [batch_size x output_size] containing the generated outputs.
                  state: The state of each decoder cell in the final time-step.
                    It is a 2D Tensor of shape [batch_size x cell.state_size].
              """
              with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
                enc_cell = copy.deepcopy(cell)
                _, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)
                if feed_previous:
                    return _rnn_decoder(decoder_inputs, enc_state, cell, _loop_function)
                else:
                    return _rnn_decoder(decoder_inputs, enc_state, cell)

            def _loop_function(prev, _):
              '''Naive implementation of loop function for _rnn_decoder. Transform prev from
              dimension [batch_size x hidden_dim] to [batch_size x output_dim], which will be
              used as decoder input of next time step '''
              return tf.matmul(prev, weights['out']) + biases['out']

            dec_outputs, dec_memory = _basic_rnn_seq2seq(
                enc_inp,
                dec_inp,
                cell,
                feed_previous = feed_previous
            )

            reshaped_outputs = [tf.matmul(i, weights['out']) + biases['out'] for i in dec_outputs]

        # Training loss and optimizer
        with tf.variable_scope('Loss'):
            # L2 loss
            output_loss = 0
            for _y, _Y in zip(reshaped_outputs, target_seq):
                output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))

            # L2 regularization for weights and biases
            reg_loss = 0
            for tf_var in tf.trainable_variables():
                if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                    reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

            loss = output_loss + lambda_l2_reg * reg_loss

        with tf.variable_scope('Optimizer'):
            optimizer = tf.contrib.layers.optimize_loss(
                    loss=loss,
                    learning_rate=learning_rate,
                    global_step=global_step,
                    optimizer='SGD',
                    clip_gradients=GRADIENT_CLIPPING)

        saver = tf.train.Saver

        return dict(
            enc_inp = enc_inp,
            target_seq = target_seq,
            train_op = optimizer,
            loss=loss,
            saver = saver,
            reshaped_outputs = reshaped_outputs,
            )







    ###RNN Architecture 5-LSTM peepholes with SGD optimizer
    def build_graph5(feed_previous = False):

        tf.reset_default_graph()

        global_step = tf.Variable(
                      initial_value=0,
                      name="global_step",
                      trainable=False,
                      collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        weights = {
            'out': tf.get_variable('Weights_out', \
                                   shape = [hidden_dim, output_dim], \
                                   dtype = tf.float32, \
                                   initializer = tf.truncated_normal_initializer()),
        }
        biases = {
            'out': tf.get_variable('Biases_out', \
                                   shape = [output_dim], \
                                   dtype = tf.float32, \
                                   initializer = tf.constant_initializer(0.)),
        }

        with tf.variable_scope('Seq2seq'):
            # Encoder: inputs
            enc_inp = [
                tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
                   for t in range(input_seq_len)
            ]

            # Decoder: target outputs
            target_seq = [
                tf.placeholder(tf.float32, shape=(None, output_dim), name="y".format(t))
                  for t in range(output_seq_len)
            ]

            # Give a "GO" token to the decoder.
            # If dec_inp are fed into decoder as inputs, this is 'guided' training; otherwise only the
            # first element will be fed as decoder input which is then 'un-guided'
            dec_inp = [ tf.zeros_like(target_seq[0], dtype=tf.float32, name="GO") ] + target_seq[:-1]

            with tf.variable_scope('LSTMCell'):
                cells = []
                for i in range(num_stacked_layers):
                    with tf.variable_scope('RNN_{}'.format(i)):
                        cells.append(tf.contrib.rnn.LSTMCell(hidden_dim, use_peepholes= True))
                cell = tf.contrib.rnn.MultiRNNCell(cells)

            def _rnn_decoder(decoder_inputs,
                            initial_state,
                            cell,
                            loop_function=None,
                            scope=None):
              """RNN decoder for the sequence-to-sequence model.
              Args:
                decoder_inputs: A list of 2D Tensors [batch_size x input_size].
                initial_state: 2D Tensor with shape [batch_size x cell.state_size].
                cell: rnn_cell.RNNCell defining the cell function and size.
                loop_function: If not None, this function will be applied to the i-th output
                  in order to generate the i+1-st input, and decoder_inputs will be ignored,
                  except for the first element ("GO" symbol). This can be used for decoding,
                  but also for training to emulate http://arxiv.org/abs/1506.03099.
                  Signature -- loop_function(prev, i) = next
                    * prev is a 2D Tensor of shape [batch_size x output_size],
                    * i is an integer, the step number (when advanced control is needed),
                    * next is a 2D Tensor of shape [batch_size x input_size].
                scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
              Returns:
                A tuple of the form (outputs, state), where:
                  outputs: A list of the same length as decoder_inputs of 2D Tensors with
                    shape [batch_size x output_size] containing generated outputs.
                  state: The state of each cell at the final time-step.
                    It is a 2D Tensor of shape [batch_size x cell.state_size].
                    (Note that in some cases, like basic RNN cell or GRU cell, outputs and
                     states can be the same. They are different for LSTM cells though.)
              """
              with variable_scope.variable_scope(scope or "rnn_decoder"):
                state = initial_state
                outputs = []
                prev = None
                for i, inp in enumerate(decoder_inputs):
                  if loop_function is not None and prev is not None:
                    with variable_scope.variable_scope("loop_function", reuse=True):
                      inp = loop_function(prev, i)
                  if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()
                  output, state = cell(inp, state)
                  outputs.append(output)
                  if loop_function is not None:
                    prev = output
              return outputs, state

            def _basic_rnn_seq2seq(encoder_inputs,
                                  decoder_inputs,
                                  cell,
                                  feed_previous,
                                  dtype=dtypes.float32,
                                  scope=None):
              """Basic RNN sequence-to-sequence model.
              This model first runs an RNN to encode encoder_inputs into a state vector,
              then runs decoder, initialized with the last encoder state, on decoder_inputs.
              Encoder and decoder use the same RNN cell type, but don't share parameters.
              Args:
                encoder_inputs: A list of 2D Tensors [batch_size x input_size].
                decoder_inputs: A list of 2D Tensors [batch_size x input_size].
                feed_previous: Boolean; if True, only the first of decoder_inputs will be
                  used (the "GO" symbol), all other inputs will be generated by the previous
                  decoder output using _loop_function below. If False, decoder_inputs are used
                  as given (the standard decoder case).
                dtype: The dtype of the initial state of the RNN cell (default: tf.float32).
                scope: VariableScope for the created subgraph; default: "basic_rnn_seq2seq".
              Returns:
                A tuple of the form (outputs, state), where:
                  outputs: A list of the same length as decoder_inputs of 2D Tensors with
                    shape [batch_size x output_size] containing the generated outputs.
                  state: The state of each decoder cell in the final time-step.
                    It is a 2D Tensor of shape [batch_size x cell.state_size].
              """
              with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
                enc_cell = copy.deepcopy(cell)
                _, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)
                if feed_previous:
                    return _rnn_decoder(decoder_inputs, enc_state, cell, _loop_function)
                else:
                    return _rnn_decoder(decoder_inputs, enc_state, cell)

            def _loop_function(prev, _):
              '''Naive implementation of loop function for _rnn_decoder. Transform prev from
              dimension [batch_size x hidden_dim] to [batch_size x output_dim], which will be
              used as decoder input of next time step '''
              return tf.matmul(prev, weights['out']) + biases['out']

            dec_outputs, dec_memory = _basic_rnn_seq2seq(
                enc_inp,
                dec_inp,
                cell,
                feed_previous = feed_previous
            )

            reshaped_outputs = [tf.matmul(i, weights['out']) + biases['out'] for i in dec_outputs]

        # Training loss and optimizer
        with tf.variable_scope('Loss'):
            # L2 loss
            output_loss = 0
            for _y, _Y in zip(reshaped_outputs, target_seq):
                output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))

            # L2 regularization for weights and biases
            reg_loss = 0
            for tf_var in tf.trainable_variables():
                if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                    reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

            loss = output_loss + lambda_l2_reg * reg_loss

        with tf.variable_scope('Optimizer'):
            optimizer = tf.contrib.layers.optimize_loss(
                    loss=loss,
                    learning_rate=learning_rate,
                    global_step=global_step,
                    optimizer='SGD',
                    clip_gradients=GRADIENT_CLIPPING)

        saver = tf.train.Saver

        return dict(
            enc_inp = enc_inp,
            target_seq = target_seq,
            train_op = optimizer,
            loss=loss,
            saver = saver,
            reshaped_outputs = reshaped_outputs,
            )







    ###RNN Architecture 6-GRU with SGD optimizer
    def build_graph6(feed_previous = False):

        tf.reset_default_graph()

        global_step = tf.Variable(
                      initial_value=0,
                      name="global_step",
                      trainable=False,
                      collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        weights = {
            'out': tf.get_variable('Weights_out', \
                                   shape = [hidden_dim, output_dim], \
                                   dtype = tf.float32, \
                                   initializer = tf.truncated_normal_initializer()),
        }
        biases = {
            'out': tf.get_variable('Biases_out', \
                                   shape = [output_dim], \
                                   dtype = tf.float32, \
                                   initializer = tf.constant_initializer(0.)),
        }

        with tf.variable_scope('Seq2seq'):
            # Encoder: inputs
            enc_inp = [
                tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
                   for t in range(input_seq_len)
            ]

            # Decoder: target outputs
            target_seq = [
                tf.placeholder(tf.float32, shape=(None, output_dim), name="y".format(t))
                  for t in range(output_seq_len)
            ]

            # Give a "GO" token to the decoder.
            # If dec_inp are fed into decoder as inputs, this is 'guided' training; otherwise only the
            # first element will be fed as decoder input which is then 'un-guided'
            dec_inp = [ tf.zeros_like(target_seq[0], dtype=tf.float32, name="GO") ] + target_seq[:-1]

            with tf.variable_scope('GRUCell'):
                cells = []
                for i in range(num_stacked_layers):
                    with tf.variable_scope('RNN_{}'.format(i)):
                        cells.append(tf.contrib.rnn.GRUCell(hidden_dim))
                cell = tf.contrib.rnn.MultiRNNCell(cells)

            def _rnn_decoder(decoder_inputs,
                            initial_state,
                            cell,
                            loop_function=None,
                            scope=None):
              """RNN decoder for the sequence-to-sequence model.
              Args:
                decoder_inputs: A list of 2D Tensors [batch_size x input_size].
                initial_state: 2D Tensor with shape [batch_size x cell.state_size].
                cell: rnn_cell.RNNCell defining the cell function and size.
                loop_function: If not None, this function will be applied to the i-th output
                  in order to generate the i+1-st input, and decoder_inputs will be ignored,
                  except for the first element ("GO" symbol). This can be used for decoding,
                  but also for training to emulate http://arxiv.org/abs/1506.03099.
                  Signature -- loop_function(prev, i) = next
                    * prev is a 2D Tensor of shape [batch_size x output_size],
                    * i is an integer, the step number (when advanced control is needed),
                    * next is a 2D Tensor of shape [batch_size x input_size].
                scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
              Returns:
                A tuple of the form (outputs, state), where:
                  outputs: A list of the same length as decoder_inputs of 2D Tensors with
                    shape [batch_size x output_size] containing generated outputs.
                  state: The state of each cell at the final time-step.
                    It is a 2D Tensor of shape [batch_size x cell.state_size].
                    (Note that in some cases, like basic RNN cell or GRU cell, outputs and
                     states can be the same. They are different for LSTM cells though.)
              """
              with variable_scope.variable_scope(scope or "rnn_decoder"):
                state = initial_state
                outputs = []
                prev = None
                for i, inp in enumerate(decoder_inputs):
                  if loop_function is not None and prev is not None:
                    with variable_scope.variable_scope("loop_function", reuse=True):
                      inp = loop_function(prev, i)
                  if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()
                  output, state = cell(inp, state)
                  outputs.append(output)
                  if loop_function is not None:
                    prev = output
              return outputs, state

            def _basic_rnn_seq2seq(encoder_inputs,
                                  decoder_inputs,
                                  cell,
                                  feed_previous,
                                  dtype=dtypes.float32,
                                  scope=None):
              """Basic RNN sequence-to-sequence model.
              This model first runs an RNN to encode encoder_inputs into a state vector,
              then runs decoder, initialized with the last encoder state, on decoder_inputs.
              Encoder and decoder use the same RNN cell type, but don't share parameters.
              Args:
                encoder_inputs: A list of 2D Tensors [batch_size x input_size].
                decoder_inputs: A list of 2D Tensors [batch_size x input_size].
                feed_previous: Boolean; if True, only the first of decoder_inputs will be
                  used (the "GO" symbol), all other inputs will be generated by the previous
                  decoder output using _loop_function below. If False, decoder_inputs are used
                  as given (the standard decoder case).
                dtype: The dtype of the initial state of the RNN cell (default: tf.float32).
                scope: VariableScope for the created subgraph; default: "basic_rnn_seq2seq".
              Returns:
                A tuple of the form (outputs, state), where:
                  outputs: A list of the same length as decoder_inputs of 2D Tensors with
                    shape [batch_size x output_size] containing the generated outputs.
                  state: The state of each decoder cell in the final time-step.
                    It is a 2D Tensor of shape [batch_size x cell.state_size].
              """
              with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
                enc_cell = copy.deepcopy(cell)
                _, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)
                if feed_previous:
                    return _rnn_decoder(decoder_inputs, enc_state, cell, _loop_function)
                else:
                    return _rnn_decoder(decoder_inputs, enc_state, cell)

            def _loop_function(prev, _):
              '''Naive implementation of loop function for _rnn_decoder. Transform prev from
              dimension [batch_size x hidden_dim] to [batch_size x output_dim], which will be
              used as decoder input of next time step '''
              return tf.matmul(prev, weights['out']) + biases['out']

            dec_outputs, dec_memory = _basic_rnn_seq2seq(
                enc_inp,
                dec_inp,
                cell,
                feed_previous = feed_previous
            )

            reshaped_outputs = [tf.matmul(i, weights['out']) + biases['out'] for i in dec_outputs]

        # Training loss and optimizer
        with tf.variable_scope('Loss'):
            # L2 loss
            output_loss = 0
            for _y, _Y in zip(reshaped_outputs, target_seq):
                output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))

            # L2 regularization for weights and biases
            reg_loss = 0
            for tf_var in tf.trainable_variables():
                if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                    reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

            loss = output_loss + lambda_l2_reg * reg_loss

        with tf.variable_scope('Optimizer'):
            optimizer = tf.contrib.layers.optimize_loss(
                    loss=loss,
                    learning_rate=learning_rate,
                    global_step=global_step,
                    optimizer='SGD',
                    clip_gradients=GRADIENT_CLIPPING)

        saver = tf.train.Saver

        return dict(
            enc_inp = enc_inp,
            target_seq = target_seq,
            train_op = optimizer,
            loss=loss,
            saver = saver,
            reshaped_outputs = reshaped_outputs,
            )






    #################################################################################################








    #Training parameters

    epochs = 5000
    #batch_size = 16
    batch_size = 6# if not specified, it will use batch size specified in the method!!
    KEEP_RATE = 0.7
    train_losses = []
    val_losses = []



    #Create  repository to hold all the training data that will be used by all the RNN
    #architectures

    xdata=[]
    ydata=[]
    for i in range(epochs):
        xsample,ysample= generate_train_samples(batch_size=batch_size)
        xdata.append(xsample)
        ydata.append(ysample)


    #create an array an by doing so removing the commas from the list structure
    #these array contains all our needed training samples.
    xarray=np.asarray(xdata)
    yarray=np.asarray(ydata)

    #Quality Control to encure the last data from the loop and the array are similar
    ##print ("x is",x)
    ##print ("xarray is",xarray[4999])
    ####



    ########################################################################################################
    #start the RNN learning sessions

    ### I) ADAM OPTIMIZER

    #Session 1) LSTM standard with Adam optimizer
    rnn_model = build_graph(feed_previous=False)
    saver = tf.train.Saver()

    start_train = time.time()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)
        print("Training losses: ")
        for i in range(epochs):
            batch_input =xarray[i]
            batch_output =yarray[i]

            feed_dict = {rnn_model['enc_inp'][t]: batch_input[:,t] for t in range(input_seq_len)}
            feed_dict.update({rnn_model['target_seq'][t]: batch_output[:,t] for t in range(output_seq_len)})
            _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
            print("loss value LSTM STD (ADAM) is =",loss_t)


        temp_saver = rnn_model['saver']()
        save_path = temp_saver.save(sess, os.path.join('./', 'multivariate_ts_pollution_case'))

    print("Checkpoint saved at: ", save_path)

    end_train = time.time()

    #Session 2) LSTM peepholes with Adam optimizer
    rnn_model2 = build_graph2(feed_previous=False)

    saver2 = tf.train.Saver()

    start_train2 = time.time()
    init2 = tf.global_variables_initializer()

    with tf.Session() as sess2:

        sess2.run(init2)


        print("Training losses: ")
        for i in range(epochs):
            batch_input =xarray[i]
            batch_output =yarray[i]


            feed_dict2 = {rnn_model2['enc_inp'][t]: batch_input[:,t] for t in range(input_seq_len)}
            feed_dict2.update({rnn_model2['target_seq'][t]: batch_output[:,t] for t in range(output_seq_len)})
            _, loss_t2 = sess2.run([rnn_model2['train_op'], rnn_model2['loss']], feed_dict2)
            print("loss value LSTM-Peephole(ADAM) is =",loss_t2)


        temp_saver2 = rnn_model2['saver']()
        save_path2 = temp_saver2.save(sess2, os.path.join('./', 'multivariate_ts_pollution_case2'))

    end_train2 = time.time()


    #Session 3) GRU with Adam optimizer
    rnn_model3 = build_graph3(feed_previous=False)

    saver3 = tf.train.Saver()

    start_train3 = time.time()
    init3 = tf.global_variables_initializer()

    with tf.Session() as sess3:

        sess3.run(init3)


        print("Training losses: ")
        for i in range(epochs):
            batch_input =xarray[i]
            batch_output =yarray[i]


            feed_dict3 = {rnn_model3['enc_inp'][t]: batch_input[:,t] for t in range(input_seq_len)}
            feed_dict3.update({rnn_model3['target_seq'][t]: batch_output[:,t] for t in range(output_seq_len)})
            _, loss_t3 = sess3.run([rnn_model3['train_op'], rnn_model3['loss']], feed_dict3)
            print("loss value GRU (ADAM) is =",loss_t3)


        temp_saver3 = rnn_model3['saver']()
        save_path3 = temp_saver3.save(sess3, os.path.join('./', 'multivariate_ts_pollution_case3'))

    end_train3 = time.time()
    ###############################################################################################################

    ### II) SGD OPTIMIZER

    #Session 4) LSTM standard with SGD optimizer
    rnn_model4 = build_graph4(feed_previous=False)
    saver4 = tf.train.Saver()

    start_train4 = time.time()
    init4 = tf.global_variables_initializer()

    with tf.Session() as sess4:

        sess4.run(init4)
        print("Training losses: ")
        for i in range(epochs):
            batch_input =xarray[i]
            batch_output =yarray[i]

            feed_dict4 = {rnn_model4['enc_inp'][t]: batch_input[:,t] for t in range(input_seq_len)}
            feed_dict4.update({rnn_model4['target_seq'][t]: batch_output[:,t] for t in range(output_seq_len)})
            _, loss_t4 = sess4.run([rnn_model4['train_op'], rnn_model4['loss']], feed_dict4)
            print("loss value LSTM STD (SGD) is =",loss_t4)


        temp_saver4 = rnn_model4['saver']()
        save_path4 = temp_saver4.save(sess4, os.path.join('./', 'multivariate_ts_pollution_case4'))

    print("Checkpoint saved at: ", save_path4)

    end_train4 = time.time()

    #Session 5) LSTM peepholes with SGD optimizer
    rnn_model5 = build_graph5(feed_previous=False)

    saver5 = tf.train.Saver()

    start_train5 = time.time()
    init5 = tf.global_variables_initializer()

    with tf.Session() as sess5:

        sess5.run(init5)


        print("Training losses: ")
        for i in range(epochs):
            batch_input =xarray[i]
            batch_output =yarray[i]


            feed_dict5 = {rnn_model5['enc_inp'][t]: batch_input[:,t] for t in range(input_seq_len)}
            feed_dict5.update({rnn_model5['target_seq'][t]: batch_output[:,t] for t in range(output_seq_len)})
            _, loss_t5 = sess5.run([rnn_model5['train_op'], rnn_model5['loss']], feed_dict5)
            print("loss value LSTM-Peephole(SGD) is =",loss_t5)


        temp_saver5 = rnn_model5['saver']()
        save_path5 = temp_saver5.save(sess5, os.path.join('./', 'multivariate_ts_pollution_case5'))

    end_train5 = time.time()


    #Session 3) GRU with SGD optimizer
    rnn_model6 = build_graph6(feed_previous=False)

    saver6 = tf.train.Saver()

    start_train6 = time.time()
    init6 = tf.global_variables_initializer()

    with tf.Session() as sess6:

        sess6.run(init6)


        print("Training losses: ")
        for i in range(epochs):
            batch_input =xarray[i]
            batch_output =yarray[i]


            feed_dict6 = {rnn_model6['enc_inp'][t]: batch_input[:,t] for t in range(input_seq_len)}
            feed_dict6.update({rnn_model6['target_seq'][t]: batch_output[:,t] for t in range(output_seq_len)})
            _, loss_t6 = sess6.run([rnn_model6['train_op'], rnn_model6['loss']], feed_dict6)
            print("loss value GRU (SGD) is =",loss_t6)


        temp_saver6 = rnn_model6['saver']()
        save_path6 = temp_saver6.save(sess6, os.path.join('./', 'multivariate_ts_pollution_case6'))

    end_train6 = time.time()





    ###############################################################################################################
    print(    ) # empty break to help in results organization
    #Training time for each session

    #Session 1
    ##print('Time taken to train LSTM Standard (ADAM) is {} minutes'.format((end_train - start_train) / 60))
    print('Time in seconds to train LSTM Standard (ADAM)is: {}'.format(end_train - start_train))
    a7=end_train - start_train

    #Session 2

    ##print('Time taken to train LSTM Peephole (ADAM) is {} minutes'.format((end_train2 - start_train2) / 60))
    print('Time in seconds to train LSTM Peephole (ADAM) is: {}'.format(end_train2 - start_train2))
    a8=end_train2 - start_train2

    #Session 3
    ##print('Time taken to train GRU (ADAM) is {} minutes'.format((end_train3 - start_train3) / 60))
    print('Time in seconds to train GRU (ADAM) is: {}'.format(end_train3 - start_train3))
    print() #to add a space to aid in results visualization
    a9=end_train3 - start_train3


    #Session 4
    ##print('Time taken to train LSTM Standard (SGD) is {} minutes'.format((end_train4 - start_train4) / 60))
    print('Time in seconds to train LSTM Standard (SGD)is: {}'.format(end_train4 - start_train4))
    a10=end_train4 - start_train4

    #Session 5
    ##print('Time taken to train LSTM Peephole (SGD) is {} minutes'.format((end_train5 - start_train5) / 60))
    print('Time in seconds to train LSTM Peephole (SGD) is: {}'.format(end_train5 - start_train5))
    a11=end_train5 - start_train5

    #Session 6
    ##print('Time taken to train GRU (SGD) is {} minutes'.format((end_train6 - start_train6) / 60))
    print('Time in seconds to train GRU (SGD) is: {}'.format(end_train6 - start_train6))
    a12=end_train6 - start_train6
    #Session 7
    #Session 8
    #Session 9

    print(    ) # empty break to help in results organization
    #################################################################################################################

    #Evaluating the accuracy of the models using test data

    #Creater a repository of test data

    test_x, test_y = generate_test_samples()

    ####################################################################################################################
    #i) Testing for ADAM optimizer

    #Model 1) LSTM standard (ADAM) evaluation

    start_test = time.time()
    rnn_model = build_graph(feed_previous=True)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)

        saver = rnn_model['saver']().restore(sess,  os.path.join('./', 'multivariate_ts_pollution_case'))

        feed_dict = {rnn_model['enc_inp'][t]: test_x[:, t, :] for t in range(input_seq_len)} # batch prediction
        feed_dict.update({rnn_model['target_seq'][t]: np.zeros([test_x.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})
        final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)

        final_preds = [np.expand_dims(pred, 1) for pred in final_preds]
        final_preds = np.concatenate(final_preds, axis = 1)
        print("Test mse LSTM STD (ADAM) is: ", np.mean((final_preds - test_y)**2))

    end_test = time.time()

    #Model 2) LSTM peephole (ADAM) evaluation
    start_test2 = time.time()
    rnn_model2 = build_graph2(feed_previous=True)
    init2 = tf.global_variables_initializer()
    with tf.Session() as sess2:

        sess2.run(init2)

        saver2 = rnn_model2['saver']().restore(sess2,  os.path.join('./', 'multivariate_ts_pollution_case2'))

        feed_dict2 = {rnn_model2['enc_inp'][t]: test_x[:, t, :] for t in range(input_seq_len)} # batch prediction
        feed_dict2.update({rnn_model2['target_seq'][t]: np.zeros([test_x.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})
        final_preds2 = sess2.run(rnn_model2['reshaped_outputs'], feed_dict2)

        final_preds2 = [np.expand_dims(pred, 1) for pred in final_preds2]
        final_preds2 = np.concatenate(final_preds2, axis = 1)
        print("Test mse LSTM peephole(ADAM) is: ", np.mean((final_preds2 - test_y)**2))

    end_test2 = time.time()

    #Model 3) GRU (ADAM) evaluation
    start_test3 = time.time()
    rnn_model3 = build_graph3(feed_previous=True)
    init3 = tf.global_variables_initializer()
    with tf.Session() as sess3:

        sess3.run(init3)

        saver3 = rnn_model3['saver']().restore(sess3,  os.path.join('./', 'multivariate_ts_pollution_case3'))

        feed_dict3 = {rnn_model3['enc_inp'][t]: test_x[:, t, :] for t in range(input_seq_len)} # batch prediction
        feed_dict3.update({rnn_model3['target_seq'][t]: np.zeros([test_x.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})
        final_preds3 = sess3.run(rnn_model3['reshaped_outputs'], feed_dict3)

        final_preds3 = [np.expand_dims(pred, 1) for pred in final_preds3]
        final_preds3 = np.concatenate(final_preds3, axis = 1)
        print("Test mse GRU(ADAM) is: ", np.mean((final_preds3 - test_y)**2))

    end_test3 = time.time()

    #############################################################################################################
    #i) Testing for SGD optimizer

    #Model 4) LSTM standard (SGD) evaluation
    print()

    start_test4 = time.time()
    rnn_model4 = build_graph4(feed_previous=True)
    init4 = tf.global_variables_initializer()
    with tf.Session() as sess4:

        sess4.run(init4)

        saver4 = rnn_model4['saver']().restore(sess4,  os.path.join('./', 'multivariate_ts_pollution_case4'))

        feed_dict4 = {rnn_model4['enc_inp'][t]: test_x[:, t, :] for t in range(input_seq_len)} # batch prediction
        feed_dict4.update({rnn_model4['target_seq'][t]: np.zeros([test_x.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})
        final_preds4 = sess4.run(rnn_model4['reshaped_outputs'], feed_dict4)

        final_preds4 = [np.expand_dims(pred, 1) for pred in final_preds4]
        final_preds4 = np.concatenate(final_preds4, axis = 1)
        print("Test mse LSTM STD (SGD) is: ", np.mean((final_preds4 - test_y)**2))

    end_test4 = time.time()

    #Model 5) LSTM peephole (SGD) evaluation
    start_test5 = time.time()
    rnn_model5 = build_graph5(feed_previous=True)
    init5 = tf.global_variables_initializer()
    with tf.Session() as sess5:

        sess5.run(init5)

        saver5 = rnn_model5['saver']().restore(sess5,  os.path.join('./', 'multivariate_ts_pollution_case5'))

        feed_dict5 = {rnn_model5['enc_inp'][t]: test_x[:, t, :] for t in range(input_seq_len)} # batch prediction
        feed_dict5.update({rnn_model5['target_seq'][t]: np.zeros([test_x.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})
        final_preds5 = sess5.run(rnn_model5['reshaped_outputs'], feed_dict5)

        final_preds5 = [np.expand_dims(pred, 1) for pred in final_preds5]
        final_preds5 = np.concatenate(final_preds5, axis = 1)
        print("Test mse LSTM peephole(SGD) is: ", np.mean((final_preds5 - test_y)**2))

    end_test5 = time.time()

    #Model 6) GRU (SGD) evaluation
    start_test6 = time.time()
    rnn_model6 = build_graph6(feed_previous=True)
    init6 = tf.global_variables_initializer()
    with tf.Session() as sess6:

        sess6.run(init6)

        saver6 = rnn_model6['saver']().restore(sess6,  os.path.join('./', 'multivariate_ts_pollution_case6'))

        feed_dict6 = {rnn_model6['enc_inp'][t]: test_x[:, t, :] for t in range(input_seq_len)} # batch prediction
        feed_dict6.update({rnn_model6['target_seq'][t]: np.zeros([test_x.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})
        final_preds6 = sess6.run(rnn_model6['reshaped_outputs'], feed_dict6)

        final_preds6 = [np.expand_dims(pred, 1) for pred in final_preds6]
        final_preds6 = np.concatenate(final_preds6, axis = 1)
        print("Test mse GRU(SGD) is: ", np.mean((final_preds6 - test_y)**2))

    end_test6 = time.time()
    #############################################################################################################


    # Evaluate time taken to test the models
    print(    ) # empty break to help in results organization

    ##print('Time taken to test LSTM STD (ADAM) is: {} minutes.'.format((end_test - start_test) / 60))
    print('Time in seconds to test LSTM STD (ADAM) is: {}'.format(end_test - start_test))
    a13=end_test - start_test

    ##print('Time taken to test LSTM peephole(ADAM) is: {} minutes.'.format((end_test3 - start_test3) / 60))
    print('Time in seconds to test LSTM peephole(ADAM) is: {}'.format(end_test2 - start_test2))
    a14=end_test2 - start_test2

    ##print('Time taken to test GRU(ADAM) is: {} minutes.'.format((end_test3 - start_test3) / 60))
    print('Time in seconds to test GRU(ADAM) is: {}'.format(end_test3 - start_test3))
    print()
    a15=end_test3 - start_test3

    ##print('Time taken to test LSTM STD (SGD) is: {} minutes.'.format((end_test4 - start_test4) / 60))
    print('Time in seconds to test LSTM STD (SGD) is: {}'.format(end_test4 - start_test4))
    a16=end_test4 - start_test4

    ##print('Time taken to test LSTM peephole(SGD) is: {} minutes.'.format((end_test5 - start_test5) / 60))
    print('Time in seconds to test LSTM peephole(SGD) is: {}'.format(end_test5 - start_test5))
    a17=end_test5 - start_test5

    ##print('Time taken to test GRU(ADAM) is: {} minutes.'.format((end_test6 - start_test6) / 60))
    print('Time in seconds to test GRU(SGD) is: {}'.format(end_test6 - start_test6))
    print()
    a18=end_test6 - start_test6
    ##############################################################################################################



    #Visualizing the evaluation results.



    # ### Unscale the predictions'''i.e Remove the z factor that had been used earlier
    # make into one array

    #For Model 1) LSTM standard (ADAM) evaluation

    dim1, dim2 = final_preds.shape[0], final_preds.shape[1]
    preds_flattened = final_preds.reshape(dim1*dim2, 1)
    unscaled_yhat = pd.DataFrame(preds_flattened, columns=['p2hr']).apply(lambda x: (x*y_std) + y_mean)
    yhat_inv = unscaled_yhat.values

    test_y_flattened = test_y.reshape(dim1*dim2, 1)
    unscaled_y = pd.DataFrame(test_y_flattened, columns=['p2hr']).apply(lambda x: (x*y_std) + y_mean)
    y_inv = unscaled_y.values

    pd.concat((unscaled_y,unscaled_yhat),axis=1)

    #For Model 2) LSTM Peephole (ADAM) evaluation

    dim12, dim22 = final_preds2.shape[0], final_preds2.shape[1]
    preds_flattened2 = final_preds2.reshape(dim12*dim22, 1)
    unscaled_yhat2 = pd.DataFrame(preds_flattened2, columns=['p2hr']).apply(lambda x: (x*y_std) + y_mean)
    yhat_inv2 = unscaled_yhat2.values

    test_y_flattened2 = test_y.reshape(dim12*dim22, 1)
    unscaled_y2 = pd.DataFrame(test_y_flattened2, columns=['p2hr']).apply(lambda x: (x*y_std) + y_mean)
    y_inv2 = unscaled_y2.values

    pd.concat((unscaled_y2,unscaled_yhat2),axis=1)

    #For Model 3) GRU (ADAM) evaluation
    dim13, dim23 = final_preds3.shape[0], final_preds3.shape[1]
    preds_flattened3 = final_preds3.reshape(dim13*dim23, 1)
    unscaled_yhat3 = pd.DataFrame(preds_flattened3, columns=['p2hr']).apply(lambda x: (x*y_std) + y_mean)
    yhat_inv3 = unscaled_yhat3.values

    test_y_flattened3 = test_y.reshape(dim13*dim23, 1)
    unscaled_y3 = pd.DataFrame(test_y_flattened3, columns=['p2hr']).apply(lambda x: (x*y_std) + y_mean)
    y_inv3= unscaled_y3.values

    pd.concat((unscaled_y3,unscaled_yhat3),axis=1)


    #For Model 4) LSTM std (SGD) evaluation

    dim14, dim24 = final_preds4.shape[0], final_preds4.shape[1]
    preds_flattened4 = final_preds4.reshape(dim14*dim24, 1)
    unscaled_yhat4 = pd.DataFrame(preds_flattened4, columns=['p2hr']).apply(lambda x: (x*y_std) + y_mean)
    yhat_inv4 = unscaled_yhat4.values

    test_y_flattened4 = test_y.reshape(dim14*dim24, 1)
    unscaled_y4 = pd.DataFrame(test_y_flattened4, columns=['p2hr']).apply(lambda x: (x*y_std) + y_mean)
    y_inv4 = unscaled_y4.values

    pd.concat((unscaled_y4,unscaled_yhat4),axis=1)

    #For Model 5) LSTM peephole (SGD) evaluation

    dim15, dim25 = final_preds5.shape[0], final_preds5.shape[1]
    preds_flattened5 = final_preds5.reshape(dim15*dim25, 1)
    unscaled_yhat5 = pd.DataFrame(preds_flattened5, columns=['p2hr']).apply(lambda x: (x*y_std) + y_mean)
    yhat_inv5 = unscaled_yhat5.values

    test_y_flattened5 = test_y.reshape(dim15*dim25, 1)
    unscaled_y5 = pd.DataFrame(test_y_flattened5, columns=['p2hr']).apply(lambda x: (x*y_std) + y_mean)
    y_inv5 = unscaled_y5.values

    pd.concat((unscaled_y5,unscaled_yhat5),axis=1)

    #For Model 6) GRU (SGD) evaluation

    dim16, dim26 = final_preds6.shape[0], final_preds6.shape[1]
    preds_flattened6 = final_preds6.reshape(dim16*dim26, 1)
    unscaled_yhat6 = pd.DataFrame(preds_flattened6, columns=['p2hr']).apply(lambda x: (x*y_std) + y_mean)
    yhat_inv6 = unscaled_yhat6.values

    test_y_flattened6 = test_y.reshape(dim16*dim26, 1)
    unscaled_y6 = pd.DataFrame(test_y_flattened6, columns=['p2hr']).apply(lambda x: (x*y_std) + y_mean)
    y_inv6 = unscaled_y6.values

    pd.concat((unscaled_y6,unscaled_yhat6),axis=1)

    # ### Calculate RMSE and Variance score also known as R^2 (regression score function)
    mse = np.mean((yhat_inv - y_inv)**2)
    rmse = np.sqrt(mse)

    mse2 = np.mean((yhat_inv2 - y_inv2)**2)
    rmse2 = np.sqrt(mse2)

    mse3 = np.mean((yhat_inv3 - y_inv3)**2)
    rmse3 = np.sqrt(mse3)

    mse4 = np.mean((yhat_inv4 - y_inv4)**2)
    rmse4 = np.sqrt(mse4)

    mse5 = np.mean((yhat_inv5 - y_inv5)**2)
    rmse5 = np.sqrt(mse5)

    mse6 = np.mean((yhat_inv6 - y_inv6)**2)
    rmse6 = np.sqrt(mse6)

    print(    ) # empty break to help in results organization

    print('Root Mean Squared Error for LSTM STD (ADAM): {:.4f}'.format(rmse))
    print('Variance score for LSTM STD (ADAM): {:2f}'.format(r2_score(y_inv, yhat_inv)))
    a1=r2_score(y_inv, yhat_inv)
    a19=rmse

    print(    ) # empty break to help in results organization

    print('Root Mean Squared Error for LSTM peephole (ADAM): {:.4f}'.format(rmse2))
    print('Variance score for LSTM peephole (ADAM): {:2f}'.format(r2_score(y_inv2, yhat_inv2)))
    a2=r2_score(y_inv2, yhat_inv2)
    a20=rmse2

    print(    ) # empty break to help in results organization

    print('Root Mean Squared Error for GRU (ADAM): {:.4f}'.format(rmse3))
    print('Variance score for GRU (ADAM): {:2f}'.format(r2_score(y_inv3, yhat_inv3)))
    a3=r2_score(y_inv3, yhat_inv3)
    a21=rmse3

    print(    ) # empty break to help in results organization
    print(    ) # empty break to help in results organization

    print('Root Mean Squared Error for LSTM STD (SGD): {:.4f}'.format(rmse4))
    print('Variance score for LSTM STD (SGD): {:2f}'.format(r2_score(y_inv4, yhat_inv4)))
    a4=r2_score(y_inv4, yhat_inv4)
    a22=rmse4

    print(    ) # empty break to help in results organization

    print('Root Mean Squared Error for LSTM peephole (SGD): {:.4f}'.format(rmse5))
    print('Variance score for LSTM peephole (SGD): {:2f}'.format(r2_score(y_inv5, yhat_inv5)))
    a5=r2_score(y_inv5, yhat_inv5)
    a23=rmse5

    print(    ) # empty break to help in results organization

    print('Root Mean Squared Error for GRU (SGD): {:.4f}'.format(rmse6))
    print('Variance score for GRU (SGD): {:2f}'.format(r2_score(y_inv6, yhat_inv6)))
    a6=r2_score(y_inv6, yhat_inv6)
    a24=rmse6

    ###plot the graph
    def plot_test(final_preds, test_y):
        fig, ax = plt.subplots(figsize=(17,8))
        ax.set_title("Test Predictions vs. Actual For November-LSTM STD(ADAM)")
        ax.plot(final_preds, color = 'red', label = 'predicted')
        ax.plot(test_y, color = 'green', label = 'actual')
        plt.legend(loc="upper left")
        plt.show()

    def plot_test2(final_preds, test_y):
        fig, ax = plt.subplots(figsize=(17,8))
        ax.set_title("Test Predictions vs. Actual For November-LSTM Peephole(ADAM)")
        ax.plot(final_preds, color = 'red', label = 'predicted')
        ax.plot(test_y, color = 'green', label = 'actual')
        plt.legend(loc="upper left")
        plt.show()

    def plot_test3(final_preds, test_y):
        fig, ax = plt.subplots(figsize=(17,8))
        ax.set_title("Test Predictions vs. Actual For November-GRU(ADAM)")
        ax.plot(final_preds, color = 'red', label = 'predicted')
        ax.plot(test_y, color = 'green', label = 'actual')
        plt.legend(loc="upper left")
        plt.show()

    def plot_test4(final_preds, test_y):
        fig, ax = plt.subplots(figsize=(17,8))
        ax.set_title("Test Predictions vs. Actual For November-LSTM STD(SGD)")
        ax.plot(final_preds, color = 'red', label = 'predicted')
        ax.plot(test_y, color = 'green', label = 'actual')
        plt.legend(loc="upper left")
        plt.show()

    def plot_test5(final_preds, test_y):
        fig, ax = plt.subplots(figsize=(17,8))
        ax.set_title("Test Predictions vs. Actual For November-LSTM Peephole(SGD)")
        ax.plot(final_preds, color = 'red', label = 'predicted')
        ax.plot(test_y, color = 'green', label = 'actual')
        plt.legend(loc="upper left")
        plt.show()

    def plot_test6(final_preds, test_y):
        fig, ax = plt.subplots(figsize=(17,8))
        ax.set_title("Test Predictions vs. Actual For November-GRU(SGD)")
        ax.plot(final_preds, color = 'red', label = 'predicted')
        ax.plot(test_y, color = 'green', label = 'actual')
        plt.legend(loc="upper left")
        plt.show()

    vscor1.append(a1)
    vscor2.append(a2)
    vscor3.append(a3)
    vscor4.append(a4)
    vscor5.append(a5)
    vscor6.append(a6)

    trscor1.append(a7)
    trscor2.append(a8)
    trscor3.append(a9)
    trscor4.append(a10)
    trscor5.append(a11)
    trscor6.append(a12)


    tescor1.append(a13)
    tescor2.append(a14)
    tescor3.append(a15)
    tescor4.append(a16)
    tescor5.append(a17)
    tescor6.append(a18)


    ramse1.append(a19)
    ramse2.append(a20)
    ramse3.append(a21)
    ramse4.append(a22)
    ramse5.append(a23)
    ramse6.append(a24)


avg1=mean(vscor1)
avg2=mean(vscor2)
avg3=mean(vscor3)
avg4=mean(vscor4)
avg5=mean(vscor5)
avg6=mean(vscor6)

avg7=mean(trscor1)
avg8=mean(trscor2)
avg9=mean(trscor3)
avg10=mean(trscor4)
avg11=mean(trscor5)
avg12=mean(trscor6)

avg13=mean(tescor1)
avg14=mean(tescor2)
avg15=mean(tescor3)
avg16=mean(tescor4)
avg17=mean(tescor5)
avg18=mean(tescor6)

avg19=mean(ramse1)
avg20=mean(ramse2)
avg21=mean(ramse3)
avg22=mean(ramse4)
avg23=mean(ramse5)
avg24=mean(ramse6)


print()
print("Finding average time taken to TRAIN for each of the 6 architectures after 100 trials")
print()
print("Lstmstd(adam) was trained for an avg= ",avg7)
print("Lstmpeephole(adam) was trained for an avg= ",avg8)
print("GRU(adam) was trained for an avg =",avg9)
print()
print("Lstmstd(SGD) was trained for an avg =",avg10)
print("Lstmpeephole(SGD) was trained for an avg =",avg11)
print("GRU(SGD) was trained for an avg =",avg12)

print()
print("Finding average time taken to TEST for each of the 6 architectures after 100 trials")
print()
print("Lstmstd(adam) was tested for an avg= ",avg13)
print("Lstmpeephole(adam) was tested for an avg= ",avg14)
print("GRU(adam) was tested for an avg =",avg15)
print()
print("Lstmstd(SGD) was tested for an avg =",avg16)
print("Lstmpeephole(SGD) was tested for an avg avg =",avg17)
print("GRU(SGD) was tested for an avg =",avg18)


print()
print("Finding of the R^2 score of 6 architectures to use after 100 trials")
print()
print("Lstmstd(adam) avg= ",avg1)
print("Lstmpeephole(adam) avg= ",avg2)
print("GRU(adam) avg =",avg3)
print()
print("Lstmstd(SGD) avg =",avg4)
print("Lstmpeephole(SGD) avg =",avg5)
print("GRU(SGD) avg =",avg6)


print()
print("Finding RMSE of 6 architectures to use after 100 trials")
print()
print("Lstmstd(adam) avg= ",avg19)
print("Lstmpeephole(adam) avg= ",avg20)
print("GRU(adam) avg =",avg21)
print()
print("Lstmstd(SGD) avg =",avg22)
print("Lstmpeephole(SGD) avg =",avg23)
print("GRU(SGD) avg =",avg24)

    ##
    ##plot_test(yhat_inv[:256,], y_inv[:256,])
    ##
    ##plot_test2(yhat_inv2[:256,], y_inv2[:256,])
    ##
    ##plot_test3(yhat_inv3[:256,], y_inv3[:256,])
    ##
    ##plot_test4(yhat_inv3[:256,], y_inv3[:256,])
    ##
    ##plot_test5(yhat_inv3[:256,], y_inv3[:256,])
    ##
    ##plot_test6(yhat_inv3[:256,], y_inv3[:256,])