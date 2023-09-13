import tensorflow as tf
import os 
import random 
import numpy as np

from tsl.dataloader.batch_on_time  import DataLoader
from tsl.transformers.patch import PatchTST
from tsl.utils.utils import seed_everything


seed_everything(2000)

# example settings 
embed_dim = 16
source_seq_len = 336 #512
target_seq_len = 96
pred_len = 96
target_cols=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
n_targets = len(target_cols)

MAX_EPOCHS = 100

# get data path
file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(file_path)
data_path = os.path.join(root_path, "datasets", "ETT-small", "ETTh1")

ts_file = 'ts.joblib'
 
dataloader = DataLoader(
        data_path,
        ts_file,
        num_cov_global_file=None,
        cat_cov_global_file=None,
        num_cov_local_variant_file=[],
        cat_cov_local_variant_file=[],
        num_cov_local_invariant_file=[],
        cat_cov_local_invariant_file=[],
        num_cov_local_variant_names=[],
        cat_cov_local_variant_names=[],
        target_cols=target_cols,
        train_range=(0, 24*30*12),
        val_range=(24*30*12, 24*30*16),
        test_range=(24*30*16, 24*30*20),
        hist_len=source_seq_len,
        token_len=target_seq_len-pred_len,
        pred_len=pred_len,
        batch_size=128,
        freq='H',
        normalize=True,
        use_time_features=False,
        use_holiday=False,
        use_holiday_distance=False,
        normalize_time_features=False,
        use_history_for_covariates=False
)

train_ds = dataloader.generate_dataset(mode="train", shuffle=True, seed=1)
val_ds = dataloader.generate_dataset(mode="validation", shuffle=False, seed=1)
test_ds = dataloader.generate_dataset(mode="test", shuffle=False, seed=1)

# create informer model
target_cols_index = dataloader.target_cols_index

model = PatchTST(pred_len = pred_len,
                target_cols_index = target_cols_index,
                embedding_dim = embed_dim,
                num_layers = 3,
                num_heads = 4,
                ffn_hidden_dim = 128,
                patch_size = 16,
                patch_strides = 8, 
                patch_padding = "end", 
                dropout_rate = 0.3,
                linear_head_dropout_rate=0.0)

# training settings
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

train_metrics = [tf.keras.metrics.MeanAbsoluteError()]
val_metrics = [tf.keras.metrics.MeanAbsoluteError()]
test_metrics = [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]

# train step
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = loss_fn(y, y_pred)
    
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
    loss = loss_fn(y, y_pred)
    for metric in val_metrics:
        metric.update_state(y, y_pred)
    return loss

# test step
@tf.function
def test_step(x, y):
    y_pred = model(x, training=False)
    for metric in test_metrics:
        metric.update_state(y, y_pred)
    
# main loop
MAX_EPOCHS = 100
patience = 3
wait = 0
best_val_loss = np.inf

for epoch in range(MAX_EPOCHS):
    # take a batch
    for batch in train_ds:
        enc, dec = batch
        (ts_enc, num_global_enc, cat_global_enc, 
         num_local_variant_enc, cat_local_variant_enc, 
         num_local_invariant_enc, cat_local_invariant_enc, time_features_enc) = enc
        (ts_dec, num_global_dec, cat_global_dec, 
         num_local_variant_dec, cat_local_variant_dec, 
         num_local_invariant_dec, cat_local_invariant_dec, time_features_dec) = dec
        
        loss = train_step(ts_enc, ts_dec[:, -pred_len:, :])
        
    # print loss every epoch
    print(f"Epoch {epoch+1}/{MAX_EPOCHS} training loss: {loss:.4f}, MAE: {train_metrics[0].result():.4f}")
    
    # reset train metrics
    for metric in train_metrics:
        metric.reset_states()
    
    # run validation loop
    # how to run validaiton loop without batching?
    
    for val_batch in val_ds:
        enc, dec = val_batch
        (ts_enc, num_global_enc, cat_global_enc, 
         num_local_variant_enc, cat_local_variant_enc, 
         num_local_invariant_enc, cat_local_invariant_enc, time_features_enc) = enc
        (ts_dec, num_global_dec, cat_global_dec, 
         num_local_variant_dec, cat_local_variant_dec, 
         num_local_invariant_dec, cat_local_invariant_dec, time_features_dec) = dec

        # calculate loss
        loss_val = val_step(ts_enc, ts_dec[:, -pred_len:, :])
        
        # print loss every epoch
    print(f"Epoch {epoch+1}/{MAX_EPOCHS} validation loss: {loss_val:.4f}, MAE: {val_metrics[0].result():.4f}")
    
    # reset val metrics
    for metric in val_metrics:
        metric.reset_states()
    
    ## early stopping
    wait += 1
    if loss_val < best_val_loss:
        best_val_loss = loss_val
        wait = 0
        model.save_weights("patchtst.h5")
    if wait > patience:
        print('early stopping...')
        break
    
# run test loop     
for test_batch in test_ds:
    enc, dec = test_batch
    (ts_enc, num_global_enc, cat_global_enc, 
        num_local_variant_enc, cat_local_variant_enc, 
        num_local_invariant_enc, cat_local_invariant_enc, time_features_enc) = enc
    (ts_dec, num_global_dec, cat_global_dec, 
        num_local_variant_dec, cat_local_variant_dec, 
        num_local_invariant_dec, cat_local_invariant_dec, time_features_dec) = dec
        
    test_step(ts_enc, ts_dec[:, -pred_len:, :])

# print loss every epoch
print(f"Test loss MSE: {test_metrics[0].result():.4f}, MAE: {test_metrics[1].result():.4f}")
    
# reset val metrics
for metric in test_metrics:
    metric.reset_states()

print(model.summary())