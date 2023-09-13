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

from tsl.dataloader.batch_on_ts import DataLoader
from tsl.utils.utils import seed_everything
from tsl.tide import TIDE

seed_everything(1024)

EPS=1E-06


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
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  #tf.config.experimental.set_visible_devices([], 'GPU')
  if gpus:
    try:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
      print(e)

  experiment_id = _get_random_string(8)
  logging.info('Experiment id: %s', experiment_id)


  # get data path
  file_path = os.path.dirname(os.path.abspath(__file__))
  root_path = os.path.dirname(file_path)
  data_path = os.path.join(root_path, "datasets", "ETT-small", "ETTh1")

  ts_file = 'ts.joblib'

  source_seq_len = 720
  pred_len = 96
  target_seq_len = pred_len
  target_cols=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
  cat_cov_local_invariant_file='id.joblib'


  dtl = DataLoader(
          data_path=data_path,
          ts_file=ts_file,
          num_cov_global_file=None,
          cat_cov_global_file=None,
          num_cov_local_variant_file=[],
          cat_cov_local_variant_file=[],
          num_cov_local_invariant_file=[],
          cat_cov_local_invariant_file=cat_cov_local_invariant_file,
          num_cov_local_variant_names=[],
          cat_cov_local_variant_names=[],
          target_cols=target_cols,
          train_range=(0, 24*30*12),
          val_range=(24*30*12, 24*30*16),
          test_range=(24*30*16, 24*30*20),
          hist_len=source_seq_len,
          token_len=target_seq_len-pred_len,
          pred_len=pred_len,
          batch_size=min(32, len(target_cols)),
          freq='H',
          normalize=True,
          use_time_features=True,
          use_holiday=False,
          use_holiday_distance=False,
          normalize_time_features=True,
          use_history_for_covariates=True
  )

  # Create model
  hidden_size = 256
  num_layers = 1
  decoder_output_dim = 4
  hidden_dims_time_decoder = 64
  layer_norm = True 
  dropout_rate = 0.0
  
  model = TIDE(
      pred_length=pred_len,
      hidden_dims_encoder=[hidden_size] * num_layers, 
      output_dims_encoder=[hidden_size] * num_layers, 
      hidden_dims_decoder=[hidden_size], 
      output_dims_decoder=[decoder_output_dim*pred_len], 
      hidden_dims_time_encoder=64,
      output_dims_time_encoder=4,
      hidden_dims_time_decoder=hidden_dims_time_decoder,
      cat_sizes=[1],# a fake 0 for categorical
      cat_emb_size=4,
      num_ts=len(target_cols),
      layer_norm=layer_norm,
      dropout_rate=dropout_rate,
  )

  
  # LR scheduling
  learning_rate = 1e-04
  lr_schedule = keras.optimizers.schedules.CosineDecay(
      initial_learning_rate=learning_rate,
      decay_steps=30 * dtl.train_range[1],
  )

  # loss function
  loss_fcn = keras.losses.MeanSquaredError()
  optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=1e3)

  # training
  MAX_EPOCHS = 10
  epoch = 0
  while epoch < MAX_EPOCHS:
    iterator = tqdm(dtl.generate_dataset(mode='train', shuffle=True), mininterval=2)
    for i, batch in enumerate(iterator):
      enc, dec = batch
      (ts_enc, num_global_enc, cat_global_enc, 
        num_local_variant_enc, cat_local_variant_enc, 
        num_local_invariant_enc, cat_local_invariant_enc, time_features_enc) = enc
      (ts_dec, num_global_dec, cat_global_dec, 
        num_local_variant_dec, cat_local_variant_dec, 
        num_local_invariant_dec, cat_local_invariant_dec, time_features_dec) = dec      
      
      # (B, L), (nx, L), ()
      past_data = (tf.squeeze(ts_enc), tf.transpose(time_features_enc[0,:, :], perm=(1,0)), tf.zeros((1, source_seq_len)))
      # (nx, H), () 
      future_features = (tf.transpose(time_features_dec[0,:,:], perm=(1,0)), tf.zeros((1,target_seq_len)))
      # (B,)
      tsidx = tf.squeeze(cat_local_invariant_enc) # (B, )
      # (B, H)
      targets = tf.squeeze(ts_dec)

      loss = train_step((past_data, future_features, tsidx), targets, model, optimizer, loss_fcn)
      
    # train metrics
    print('train/reg_loss:', loss, 'train/loss:', loss)
    
    # next
    epoch += 1

def main(_):
  training()
  
if __name__ == '__main__':
  app.run(main)
  