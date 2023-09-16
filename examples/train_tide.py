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


dataloader = DataLoader(
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
        use_holiday_distance=True,
        normalize_time_features=True,
        use_history_for_covariates=True
)

train_ds = dataloader.generate_dataset(mode="train", shuffle=True, seed=1)
val_ds = dataloader.generate_dataset(mode="validation", shuffle=False, seed=1)
test_ds = dataloader.generate_dataset(mode="test", shuffle=False, seed=1)

# Create model
hidden_size = 512
num_layers = 2
decoder_output_dim = 32
hidden_dims_time_encoder = 64
output_dims_time_encoder = 4
hidden_dims_time_decoder = 16
local_invariant_vocab_sizes = dataloader._cat_cov_local_invariant_sizes
local_invariant_emb_sizes = [4]*len(dataloader._cat_cov_local_invariant_sizes)
global_vocab_sizes = dataloader._cat_cov_global_sizes
global_emb_sizes = [4]*len(dataloader._cat_cov_global_sizes)
local_variant_vocab_sizes = dataloader._cat_cov_local_variant_sizes
local_variant_emb_sizes = [4]*len(dataloader._cat_cov_local_variant_sizes)

layer_norm = True 
dropout_rate = 0.5

model = TIDE(
    hist_length=source_seq_len,
    pred_length=pred_len,
    hidden_dims_encoder=[hidden_size] * num_layers, 
    output_dims_encoder=[hidden_size] * num_layers, 
    hidden_dims_decoder=[hidden_size], 
    output_dims_decoder=[decoder_output_dim*pred_len], 
    hidden_dims_time_encoder=hidden_dims_time_encoder,
    output_dims_time_encoder=output_dims_time_encoder,
    hidden_dims_time_decoder=hidden_dims_time_decoder,
    local_invariant_vocab_sizes=local_invariant_vocab_sizes,
    local_invariant_emb_sizes=local_invariant_emb_sizes,
    global_vocab_sizes=global_vocab_sizes,
    global_emb_sizes=global_emb_sizes,
    local_variant_vocab_sizes=local_variant_vocab_sizes,
    local_variant_emb_sizes=local_variant_emb_sizes,
    layer_norm=layer_norm,
    dropout_rate=dropout_rate,
)


# LR scheduling
learning_rate = 0.000984894211777642
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=learning_rate,
    decay_steps=30 * dataloader.train_range[1],
)

# loss function
loss_fcn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=1e3)

train_metrics = [keras.metrics.MeanAbsoluteError()]
val_metrics = [keras.metrics.MeanAbsoluteError()]
test_metrics = [keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]

# train step
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = loss_fcn(y, y_pred)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # update metrics
    for metric in train_metrics:
            metric.update_state(y, y_pred)
    return loss

# validation step
@tf.function
def val_step(x, y):
    y_pred = model(x, training=False)
    loss = loss_fcn(y, y_pred)
    for metric in val_metrics:
        metric.update_state(y, y_pred)
    return loss

# test step
@tf.function
def test_step(x, y):
    y_pred = model(x, training=False)
    for metric in test_metrics:
        metric.update_state(y, y_pred)

# training
# main loop
MAX_EPOCHS = 100
patience = 20
wait = 0
best_val_loss = np.inf

@tf.function
def get_mdoel_inputs(batch):
  enc, dec = batch
  (ts_enc, num_global_enc, cat_global_enc, 
    num_local_variant_enc, cat_local_variant_enc, 
    num_local_invariant_enc, cat_local_invariant_enc, time_features_enc) = enc
  (ts_dec, num_global_dec, cat_global_dec, 
    num_local_variant_dec, cat_local_variant_dec, 
    num_local_invariant_dec, cat_local_invariant_dec, time_features_dec) = dec
  
  # local invariant features: numeric and categorical
  # (B, Nlin)
  num_cov_local_invariant = num_local_invariant_enc
  # (B, Nlic)
  cat_cov_local_invariant = cat_local_invariant_enc
  
  # global features
  # (L+H, Nt) 
  time_features = tf.concat([time_features_enc[0, :, :], time_features_dec[0, :, :]], axis=0)
  # (L+H, Ngn)
  num_cov_global = None
  if len(num_global_enc.shape) > 0:
    num_cov_global = tf.concat([num_global_enc[0, :, :], num_global_dec[0, :, :]], axis=0)
  # (L+H, Ngc)
  cat_cov_global = None
  if len(cat_global_enc.shape) > 0:
    cat_cov_global = tf.concat([cat_global_enc[0, :, :], cat_global_dec[0, :, :]], axis=0)
  
  # local variant
  # (B, L+H, Nlvn)
  num_cov_local_variant = None
  if len(num_local_variant_enc.shape) > 0:
    num_cov_local_variant = tf.concat([num_local_variant_enc, num_local_variant_dec], axis=1)
  # (B, L+H, Nlvc)
  cat_cov_local_variant = None
  if len(cat_local_variant_enc.shape) > 0:
    cat_cov_local_variant = tf.concat([cat_local_variant_enc, cat_local_variant_dec], axis=1)
  
  # combine   
  inputs = (tf.squeeze(ts_enc), 
            num_cov_local_invariant, 
            cat_cov_local_invariant, 
            time_features, 
            num_cov_global,
            cat_cov_global,
            num_cov_local_variant,
            cat_cov_local_variant)
  targets = tf.squeeze(ts_dec)

  return inputs, targets

for epoch in range(MAX_EPOCHS):
    # take a batch
    for batch in tqdm(train_ds):
        inputs, targets = get_mdoel_inputs(batch)
        loss = train_step(inputs, targets)
        
    # print loss every epoch
    print(f"Epoch {epoch+1}/{MAX_EPOCHS} training loss: {loss:.4f}, MAE: {train_metrics[-1].result():.4f}")
    
    # reset train metrics
    for metric in train_metrics:
        metric.reset_states()
    
    # run validation loop
    # how to run validaiton loop without batching?
    for val_batch in tqdm(val_ds):
        inputs, targets = get_mdoel_inputs(val_batch)        
        loss_val = train_step(inputs, targets)
        
    # print validation metrics every epoch
    print(f"Epoch {epoch+1}/{MAX_EPOCHS} validation loss: {loss_val:.4f}, MAE: {val_metrics[-1].result():.4f}")
    
    # reset val metrics
    for metric in val_metrics:
        metric.reset_states()
    
    ## early stopping
    wait += 1
    if loss_val < best_val_loss:
        best_val_loss = loss_val
        wait = 0
        model.save_weights("tide.h5")
    if wait > patience:
        print('early stopping...')
        break

