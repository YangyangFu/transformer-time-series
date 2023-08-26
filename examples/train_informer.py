#%%
import tensorflow as tf
import os 
import random 
import numpy as np

from tsl.transformers.informer import DataLoader
from tsl.transformers.informer import Informer

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

set_seed(42)

# example settings 
embed_dim = 512
source_seq_len = 360
pred_len = 24
target_seq_len = 168 + pred_len
target_cols=['OT']
num_cov_cols=['OT'] #['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
n_num_covs = len(num_cov_cols)
n_targets = len(target_cols)

MAX_EPOCHS = 10

#%%
# get data path
file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(file_path)
data_path = os.path.join(root_path, "datasets", "ETT-small", "ETTh1.csv")

# create dataloader
dataloader = DataLoader(data_path=data_path,
                    target_cols=target_cols,
                    num_cov_cols=num_cov_cols,
                    train_range=(0, 24*30*12),
                    val_range=(24*30*12, 24*30*16),
                    test_range=(24*30*16, 24*30*20),
                    hist_len=source_seq_len,
                    token_len=target_seq_len-pred_len,
                    pred_len=pred_len,
                    batch_size=32,
                    use_time_features=True,
                    use_holiday=True,
                    use_holiday_distance=False,
                    )
train_ds = dataloader.generate_dataset(mode="train", shuffle=True, seed=1)
val_ds = dataloader.generate_dataset(mode="validation", shuffle=False, seed=1)
test_ds = dataloader.generate_dataset(mode="test", shuffle=False, seed=1)

#%%
# create informer model
model = Informer(output_dim=n_targets, 
                pred_len=pred_len,
                num_layers_encoder=2, 
                num_heads_encoder=8, 
                key_dim_encoder=64, 
                value_dim_encoder=64, 
                output_dim_encoder=512, 
                hidden_dim_encoder=2048, 
                factor_encoder=5,
                num_layers_decoder=1, 
                num_heads_decoder=8, 
                key_dim_decoder=64, 
                value_dim_decoder=64, 
                output_dim_decoder=512, 
                hidden_dim_decoder=2048, 
                factor_decoder=5, 
                num_cat_cov=0,
                cat_cov_embedding_size=[],
                cat_cov_embedding_dim=16,
                freq='H',
                use_holiday=True,
                dropout_rate=0.2,)
#%%
# training settings
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

train_metrics = [tf.keras.metrics.MeanAbsoluteError()]
val_metrics = [tf.keras.metrics.MeanAbsoluteError()]
test_metrics = [tf.keras.metrics.MeanAbsoluteError()]

# train step
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        x_enc, x_dec = x
        y_pred = model(x_enc, x_dec, training=True)
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
    x_enc, x_dec = x
    y_pred = model(x_enc, x_dec, training=False)
    loss = loss_fn(y, y_pred)
    for metric in val_metrics:
        metric.update_state(y, y_pred)
    return loss

# test step
@tf.function
def test_step(x, y):
    x_enc, x_dec = x
    y_pred = model(x_enc, x_dec, training=False)
    loss = loss_fn(y, y_pred)
    for metric in test_metrics:
        metric.update_state(y, y_pred)
    return loss


#%%
# main loop
MAX_EPOCHS = 8

# early stopping
patience = 3
wait = 0
best = np.Inf

# train loop
for epoch in range(MAX_EPOCHS):
    # take a batch
    for batch in train_ds:
        num_covs, cat_covs, time_enc, time_dec, target_dec = batch
        
        #try: 
        # zero for target 
        token_dec = target_dec[:, :-pred_len, :]
        zeros = tf.zeros_like(target_dec[:, -pred_len:, :])
        token_target_dec = tf.concat([token_dec, zeros], axis=1)
        
        # feed model
        x_enc = (num_covs, cat_covs, time_enc)
        x_dec = (time_dec, token_target_dec)
        
        # train step
        loss = train_step((x_enc, x_dec), target_dec[:, -pred_len:, :])
            
        #except tf.errors.OutOfRangeError:
        #    pass
        
    # print loss every epoch
    print(f"Epoch {epoch+1}/{MAX_EPOCHS} training loss: {loss:.4f}, MAE: {train_metrics[0].result():.4f}")
    
    # reset train metrics
    for metric in train_metrics:
        metric.reset_states()
    
    # run validation loop
    # how to run validaiton loop without batching?
    
    for val_batch in val_ds:
        num_covs, cat_covs, time_enc, time_dec, target_dec = val_batch
        

        # zero for target 
        token_dec = target_dec[:, :-pred_len, :]
        zeros = tf.zeros_like(target_dec[:, -pred_len:, :])
        token_target_dec = tf.concat([token_dec, zeros], axis=1)
        
        # feed model
        x_enc = (num_covs, cat_covs, time_enc)
        x_dec = (time_dec, token_target_dec)
        
        # calculate loss
        loss_val = val_step((x_enc, x_dec), target_dec[:, -pred_len:, :])
    
        # print loss every epoch
    print(f"Epoch {epoch+1}/{MAX_EPOCHS} validation loss: {loss_val:.4f}, MAE: {val_metrics[0].result():.4f}")
    
    # reset val metrics
    for metric in val_metrics:
        metric.reset_states()
        
    
    ## early stopping
    # stop the training if the validation loss does not decrease for over a certain number of epochs
    wait += 1
    if loss_val < best:
        best = loss_val
        wait = 0
    if wait >= patience:
        print('Epoch {}: early stopping'.format(epoch+1))
        break
        
#%%
for test_batch in test_ds:
    num_covs, cat_covs, time_enc, time_dec, target_dec = test_batch

    # zero for target
    token_dec = target_dec[:, :-pred_len, :]
    zeros = tf.zeros_like(target_dec[:, -pred_len:, :])
    token_target_dec = tf.concat([token_dec, zeros], axis=1)

    # feed model
    x_enc = (num_covs, cat_covs, time_enc)
    x_dec = (time_dec, token_target_dec)

    # calculate loss
    loss_test = test_step((x_enc, x_dec), target_dec[:, -pred_len:, :])

# print loss every epoch
print(f"Test loss: {loss_test:.4f}, MAE: {test_metrics[0].result():.4f}")
# reset metrics
for metric in val_metrics:
    metric.reset_states()
