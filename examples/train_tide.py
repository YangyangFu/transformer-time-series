# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main training code."""

import json
import os
import random
import string
import sys

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from tsl.tide import DataLoader
from tsl.tide import TIDE

EPS = 1e-07
FLAGS = flags.FLAGS

flags.DEFINE_integer('train_epochs', 10, 'Number of epochs to train')
flags.DEFINE_integer('patience', 40, 'Patience for early stopping')
flags.DEFINE_integer('epoch_len', None, 'number of iterations in an epoch')
flags.DEFINE_integer(
    'batch_size', 21, 'Batch size for the randomly sampled batch'
)
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')


# Non tunable flags
flags.DEFINE_string(
    'expt_dir',
    './results',
    'The name of the experiment dir',
)
flags.DEFINE_string('dataset', 'etth1', 'The name of the dataset.')
flags.DEFINE_string('datetime_col', 'date', 'Column having datetime.')
flags.DEFINE_list('num_cov_cols', None, 'Column having numerical features.')
flags.DEFINE_list('cat_cov_cols', None, 'Column having categorical features.')
flags.DEFINE_integer('hist_len', 720, 'Length of the history provided as input')
flags.DEFINE_integer('pred_len', 96, 'Length of pred len during training')
flags.DEFINE_integer('num_layers', 1, 'Number of DNN layers')
flags.DEFINE_integer('hidden_size', 256, 'Hidden size of DNN')
flags.DEFINE_integer('decoder_output_dim', 4, 'Hidden d3 of DNN')
flags.DEFINE_integer('final_decoder_hidden', 64, 'Hidden d3 of DNN')
flags.DEFINE_list('ts_cols', None, 'Columns of time-series features')
flags.DEFINE_integer(
    'random_seed', None, 'The random seed to be used for TF and numpy'
)
flags.DEFINE_bool('normalize', True, 'normalize data for training or not')
flags.DEFINE_bool('holiday', False, 'use holiday features or not')
flags.DEFINE_bool('permute', True, 'permute the order of TS in training set')
flags.DEFINE_bool('transform', False, 'Apply chronoml transform or not.')
flags.DEFINE_bool('layer_norm', True, 'Apply layer norm or not.')
flags.DEFINE_float('dropout_rate', 0.0, 'dropout rate')
flags.DEFINE_integer('num_split', 1, 'number of splits during inference.')
flags.DEFINE_integer(
    'min_num_epochs', 0, 'minimum number of epochs before early stopping'
)
flags.DEFINE_integer('gpu', 0, 'index of gpu to be used.')

DATA_DICT = {
    'ettm2': {
        'boundaries': [34560, 46080, 57600],
        'data_path': '../datasets/ETT-small/ETTm2.csv',
        'freq': '15min',
    },
    'ettm1': {
        'boundaries': [34560, 46080, 57600],
        'data_path': '../datasets/ETT-small/ETTm1.csv',
        'freq': '15min',
    },
    'etth2': {
        'boundaries': [8640, 11520, 14400],
        'data_path': '../datasets/ETT-small/ETTh2.csv',
        'freq': 'H',
    },
    'etth1': {
        'boundaries': [8640, 11520, 14400],
        'data_path': '../datasets/ETT-small/ETTh1.csv',
        'freq': 'H',
    },
    'elec': {
        'boundaries': [18413, 21044, 26304],
        'data_path': '../datasets/electricity/electricity.csv',
        'freq': 'H',
    },
    'traffic': {
        'boundaries': [12280, 14036, 17544],
        'data_path': '../datasets/traffic/traffic.csv',
        'freq': 'H',
    },
    'weather': {
        'boundaries': [36887, 42157, 52696],
        'data_path': '../datasets/weather/weather.csv',
        'freq': '10min',
    },
}

np.random.seed(1024)
tf.random.set_seed(1024)


def _get_random_string(num_chars):
  rand_str = ''.join(
      random.choice(
          string.ascii_uppercase + string.ascii_lowercase + string.digits
      )
      for _ in range(num_chars - 1)
  )
  return rand_str


 # metrics 
def mape(y_pred, y_true):
  abs_diff = np.abs(y_pred - y_true).flatten()
  abs_val = np.abs(y_true).flatten()
  idx = np.where(abs_val > EPS)
  mpe = np.mean(abs_diff[idx] / abs_val[idx])
  return mpe


def mae_loss(y_pred, y_true):
  return np.abs(y_pred - y_true).mean()


def wape(y_pred, y_true):
  abs_diff = np.abs(y_pred - y_true)
  abs_val = np.abs(y_true)
  wpe = np.sum(abs_diff) / (np.sum(abs_val) + EPS)
  return wpe


def smape(y_pred, y_true):
  abs_diff = np.abs(y_pred - y_true)
  abs_mean = (np.abs(y_true) + np.abs(y_pred)) / 2
  smpe = np.mean(abs_diff / (abs_mean + EPS))
  return smpe


def rmse(y_pred, y_true):
  return np.sqrt(np.square(y_pred - y_true).mean())


def nrmse(y_pred, y_true):
  mse = np.square(y_pred - y_true)
  return np.sqrt(mse.mean()) / np.abs(y_true).mean()


METRICS = {
    'mape': mape,
    'wape': wape,
    'smape': smape,
    'nrmse': nrmse,
    'rmse': rmse,
    'mae': mae_loss,
}

# train/validation/test function
@tf.function
def train_step(inputs, y, model, optimizer, loss_fcn):
  with tf.GradientTape() as tape:
    y_pred = model(inputs, training=True)
    loss = loss_fcn(y, y_pred)
  grads = tape.gradient(loss, model.trainable_weights)
  optimizer.apply_gradients(zip(grads, model.trainable_weights))
  return loss

@tf.function
def val_step(inputs, y, model, loss_fcn):
  y_pred = model(inputs, training=False)
  loss = loss_fcn(y, y_pred)
  return loss

def training():
  """Training TS code."""
  tf.random.set_seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)

  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_visible_devices(gpus[FLAGS.gpu], 'GPU')
  #tf.config.experimental.set_visible_devices([], 'GPU')
  if gpus:
    try:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
      print(e)

  experiment_id = _get_random_string(8)
  logging.info('Experiment id: %s', experiment_id)

  dataset = FLAGS.dataset
  data_path = DATA_DICT[dataset]['data_path']
  freq = DATA_DICT[dataset]['freq']
  boundaries = DATA_DICT[dataset]['boundaries']

  file_path = os.path.dirname(os.path.realpath(__file__))
  data_path = os.path.join(file_path, data_path)
  data_df = pd.read_csv(open(data_path, 'r'))

  if FLAGS.ts_cols:
    ts_cols = DATA_DICT[dataset]['ts_cols']
    num_cov_cols = DATA_DICT[dataset]['num_cov_cols']
    cat_cov_cols = DATA_DICT[dataset]['cat_cov_cols']
  else:
    ts_cols = [col for col in data_df.columns if col != FLAGS.datetime_col]
    num_cov_cols = None
    cat_cov_cols = None
  permute = FLAGS.permute
  dtl = DataLoader(
      data_path=data_path,
      datetime_col=FLAGS.datetime_col,
      num_cov_cols=num_cov_cols,
      cat_cov_cols=cat_cov_cols,
      ts_cols=np.array(ts_cols),
      train_range=[0, boundaries[0]],
      val_range=[boundaries[0], boundaries[1]],
      test_range=[boundaries[1], boundaries[2]],
      hist_len=FLAGS.hist_len,
      pred_len=FLAGS.pred_len,
      batch_size=min(FLAGS.batch_size, len(ts_cols)),
      freq=freq,
      normalize=FLAGS.normalize,
      epoch_len=FLAGS.epoch_len,
      holiday=FLAGS.holiday,
      permute=permute,
  )

  # Create model
  model = TIDE(
      pred_length=FLAGS.pred_len,
      hidden_dims_encoder=[FLAGS.hidden_size] * FLAGS.num_layers, 
      output_dims_encoder=[FLAGS.hidden_size] * FLAGS.num_layers, 
      hidden_dims_decoder=[FLAGS.hidden_size], 
      output_dims_decoder=[FLAGS.decoder_output_dim*FLAGS.pred_len], 
      hidden_dims_time_encoder=64,
      output_dims_time_encoder=4,
      hidden_dims_time_decoder=FLAGS.final_decoder_hidden,
      cat_sizes=dtl.cat_sizes,
      cat_emb_size=4,
      num_ts=len(ts_cols),
  #      transform=FLAGS.transform,
      layer_norm=FLAGS.layer_norm,
      dropout_rate=FLAGS.dropout_rate,
  )

  # Compute path to experiment directory
  expt_dir = os.path.join(
      FLAGS.expt_dir,
      FLAGS.dataset + '_' + str(experiment_id) + '_' + str(FLAGS.pred_len),
  )
  os.makedirs(expt_dir, exist_ok=True)


  # LR scheduling
  lr_schedule = keras.optimizers.schedules.CosineDecay(
      initial_learning_rate=FLAGS.learning_rate,
      decay_steps=30 * dtl.train_range[1],
  )

  # loss function
  loss_fcn = keras.losses.MeanSquaredError()
  optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=1e3)

  # training
  epoch = 0
  while epoch < FLAGS.train_epochs:
    iterator = tqdm(dtl.tf_dataset(mode='train'), mininterval=2)
    for i, batch in enumerate(iterator):
      past_data = batch[:3]
      future_features = batch[4:6]
      tsidx = batch[-1]
      targets = batch[3]
      loss = train_step((past_data, future_features, tsidx), targets, model, optimizer, loss_fcn)
      
    # train metrics
    print('train/reg_loss:', loss, 'train/loss:', loss)
    
    # next
    epoch += 1

def main(_):
  training()
  
if __name__ == '__main__':
  app.run(main)
  