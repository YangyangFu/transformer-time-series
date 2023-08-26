import tensorflow as tf
import os 
import random 
import numpy as np

from tsl.transformers.informer import DataLoader
from tsl.transformers.timesnet import TimesNet

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

set_seed(2023)

# example settings 
embed_dim = 16
source_seq_len = 96
target_seq_len = 96
pred_len = 96
target_cols=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
n_targets = len(target_cols)

# get data path
file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(file_path)
data_path = os.path.join(root_path, "datasets", "ETT-small", "ETTh1.csv")

# create dataloader
dataloader = DataLoader(data_path=data_path,
                    target_cols=target_cols,
                    num_cov_cols=target_cols,
                    train_range=(0, 12*30*24),
                    val_range=(12*30*24, 12*30*24+4*30*24),
                    test_range=(12*30*24+4*30*24, 12*30*24+8*30*24),
                    hist_len=source_seq_len,
                    token_len=target_seq_len-pred_len,
                    pred_len=pred_len,
                    batch_size=32,
                    use_time_features=True,
                    use_holiday=False,
                    normalize_time_features=True,
                    )
train_ds = dataloader.generate_dataset(mode="train", shuffle=True, seed=1)
val_ds = dataloader.generate_dataset(mode="validation", shuffle=False, seed=1)
test_ds = dataloader.generate_dataset(mode="test", shuffle=False, seed=1)

# create informer model
target_cols_index = dataloader.target_cols_index

model = TimesNet(target_cols_index = target_cols_index,
                pred_len = pred_len,
                hist_len = source_seq_len,
                num_layers=2,
                embedding_dim=embed_dim, 
                topk = 1,
                cov_hidden_dim = 32,
                num_kernels = 6,
                num_cat_cov=0, 
                cat_cov_embedding_size=[], 
                cat_cov_embedding_dim=4, 
                time_embedding_type="time2vec",
                use_holiday=False, 
                freq='H', 
                dropout_rate=0.1,)

# training settings
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

train_metrics = [tf.keras.metrics.MeanAbsoluteError()]
val_metrics = [tf.keras.metrics.MeanAbsoluteError()]
test_metrics = [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]

# train step
# cannot use tf.function decorator because the FFT period model has unknown shape due to the selection of different periods
#@tf.function
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
#@tf.function
def val_step(x, y):
    y_pred = model(x, training=False)
    loss = loss_fn(y, y_pred)
    for metric in val_metrics:
        metric.update_state(y, y_pred)
    return loss

# test step
#@tf.function
def test_step(x, y):
    y_pred = model(x, training=False)
    for metric in test_metrics:
        metric.update_state(y, y_pred)
    
# main loop
MAX_EPOCHS = 10
patience = 3
wait = 0
best_val_loss = np.inf

for epoch in range(MAX_EPOCHS):
    # take a batch
    for batch in train_ds:
        num_covs, cat_covs, time_enc, time_dec, target_dec = batch
        loss = train_step((num_covs, cat_covs, time_enc[:,:,:4]), target_dec[:, -pred_len:, :])
        
        # check model summary
        if epoch == 0:
            model.summary()
            # check number of parameters for different blocks in the encoder
            print(f"Number of parameters in the encoder: {model.enc.count_params()}")
            print(f"Number of parameters in the encoder blocks: {model.enc.blocks[0].count_params()}")
            print(f"Number of parameters in the encoder inception 1: {model.enc.blocks[0].conv[0].count_params()}")
            print(f"Number of parameters in the encoder inception 1: {model.enc.blocks[0].conv[1].count_params()}")
            print(f"Number of parameters in the encoder inception 1: {model.enc.blocks[0].conv[2].count_params()}")
            print(f"Number of parameters in the encoder inception 1: {model.enc.blocks[0].conv[2].kernels[0].count_params()}")
            print(f"Number of parameters in the encoder inception 1: {model.enc.blocks[0].conv[2].kernels[1].count_params()}")
            print(f"Number of parameters in the encoder inception 1: {model.enc.blocks[0].conv[2].kernels[2].count_params()}")
            print(f"Number of parameters in the encoder inception 1: {model.enc.blocks[0].conv[2].kernels[3].count_params()}")
            print(f"Number of parameters in the encoder inception 1: {model.enc.blocks[0].conv[2].kernels[4].count_params()}")
            print(f"Number of parameters in the encoder inception 1: {model.enc.blocks[0].conv[2].kernels[5].count_params()}")
            print("======================================================================================")
            print()
            print(f"Number of parameters in the encoder blocks: {model.enc.blocks[1].count_params()}")
            print(f"Number of parameters in the encoder inception 1: {model.enc.blocks[1].conv[0].count_params()}")
            print(f"Number of parameters in the encoder inception 1: {model.enc.blocks[1].conv[1].count_params()}")
            print(f"Number of parameters in the encoder inception 1: {model.enc.blocks[1].conv[2].count_params()}")
            print(f"Number of parameters in the encoder inception 1: {model.enc.blocks[1].conv[2].kernels[0].count_params()}")
            print(f"Number of parameters in the encoder inception 1: {model.enc.blocks[1].conv[2].kernels[1].count_params()}")
            print(f"Number of parameters in the encoder inception 1: {model.enc.blocks[1].conv[2].kernels[2].count_params()}")
            print(f"Number of parameters in the encoder inception 1: {model.enc.blocks[1].conv[2].kernels[3].count_params()}")
            print(f"Number of parameters in the encoder inception 1: {model.enc.blocks[1].conv[2].kernels[4].count_params()}")
            print(f"Number of parameters in the encoder inception 1: {model.enc.blocks[1].conv[2].kernels[5].count_params()}")

    # print loss every epoch
    print(f"Epoch {epoch+1}/{MAX_EPOCHS} training loss: {loss:.4f}, MAE: {train_metrics[0].result():.4f}")
    
    # reset train metrics
    for metric in train_metrics:
        metric.reset_states()
    
    # run validation loop
    # how to run validaiton loop without batching?
    
    for val_batch in val_ds:
        num_covs, cat_covs, time_enc, time_dec, target_dec = val_batch

        # calculate loss
        loss_val = val_step((num_covs, cat_covs, time_enc[:,:,:4]), target_dec[:, -pred_len:, :])
        
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
    num_covs, cat_covs, time_enc, time_dec, target_dec = test_batch
    test_step((num_covs, cat_covs, time_enc[:,:,:4]), target_dec[:, -pred_len:, :])

# print loss every epoch
print(f"Test loss MSE: {test_metrics[0].result():.4f}, MAE: {test_metrics[1].result():.4f}")
    
# reset val metrics
for metric in test_metrics:
    metric.reset_states()

print(model.summary())

