import tensorflow as tf
import os 
import random 
import numpy as np

from tsl.transformers.informer import DataLoader
from tsl.transformers.patch import PatchTST

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    tf.random.set_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

set_seed(2000)

# example settings 
embed_dim = 16
source_seq_len = 512
target_seq_len = 96
pred_len = 96
target_cols=['OT'] # ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
n_targets = len(target_cols)

MAX_EPOCHS = 100

# get data path
file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(file_path)
data_path = os.path.join(root_path, "datasets", "ETT-small", "ETTh1.csv")

# create dataloader
dataloader = DataLoader(data_path=data_path,
                    target_cols=target_cols,
                    num_cov_cols=['OT'],
                    train_range=(0, 12*30*24),
                    val_range=(12*30*24, 12*30*24+4*30*24),
                    test_range=(12*30*24+4*30*24, 12*30*24+8*30*24),
                    hist_len=source_seq_len,
                    token_len=target_seq_len-pred_len,
                    pred_len=pred_len,
                    batch_size=64,
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
        x, cat_covs, time_enc, time_dec, target_dec = batch
        loss = train_step(x, target_dec[:, -pred_len:, :])
        
    # print loss every epoch
    print(f"Epoch {epoch+1}/{MAX_EPOCHS} training loss: {loss:.4f}, MAE: {train_metrics[0].result():.4f}")
    
    # reset train metrics
    for metric in train_metrics:
        metric.reset_states()
    
    # run validation loop
    # how to run validaiton loop without batching?
    
    for val_batch in val_ds:
        x, cat_covs, time_enc, time_dec, target_dec = val_batch

        # calculate loss
        loss_val = val_step(x, target_dec[:, -pred_len:, :])
        
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
    x, cat_covs, time_enc, time_dec, target_dec = test_batch
    test_step(x, target_dec[:, -pred_len:, :])

# print loss every epoch
print(f"Test loss MSE: {test_metrics[0].result():.4f}, MAE: {test_metrics[1].result():.4f}")
    
# reset val metrics
for metric in test_metrics:
    metric.reset_states()

print(model.summary())