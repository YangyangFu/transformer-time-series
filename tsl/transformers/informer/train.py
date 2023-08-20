from dataloader import DataLoader
from model import Informer
import tensorflow as tf

embed_dim = 512
source_seq_len = 64
target_seq_len = 128
pred_len = 96
n_num_covs = 7
n_targets = 1

MAX_EPOCHS = 10

dataloader = DataLoader(data_path='ETTh1.csv',
                    target_cols=['OT'],
                    num_cov_cols=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'],
                    train_range=(0, 10000),
                    val_range=(10000, 11000),
                    test_range=(11000, 12000),
                    hist_len=source_seq_len,
                    token_len=target_seq_len-pred_len,
                    pred_len=pred_len,
                    batch_size=32,
                    )
train_ds = dataloader.generate_dataset(mode="train", shuffle=True, seed=1)
val_ds = dataloader.generate_dataset(mode="validation", shuffle=False, seed=1)
test_ds = dataloader.generate_dataset(mode="test", shuffle=False, seed=1)

# attention block
model = Informer(output_dim=n_targets, 
                pred_len=pred_len,
                num_layers_encoder=4, 
                num_heads_encoder=16, 
                key_dim_encoder=32, 
                value_dim_encoder=32, 
                output_dim_encoder=512, 
                hidden_dim_encoder=2048, 
                factor_encoder=4,
                num_layers_decoder=2, 
                num_heads_decoder=8, 
                key_dim_decoder=64, 
                value_dim_decoder=64, 
                output_dim_decoder=512, 
                hidden_dim_decoder=2048, 
                factor_decoder=4, 
                num_cat_cov=0,
                cat_cov_embedding_size=[],
                cat_cov_embedding_dim=16,
                freq='H',
                use_holiday=True,
                dropout_rate=0.1,)

# training settings
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

train_metrics = [tf.keras.metrics.MeanAbsoluteError()]
val_metrics = [tf.keras.metrics.MeanAbsoluteError()]

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
            metric.update_state(target_dec[:, pred_len:, :], y_pred)
    return loss

# validation step
@tf.function
def val_step(x, y):
    y_pred = model(x, training=False)
    loss = loss_fn(y, y_pred)
    for metric in val_metrics:
        metric.update_state(target_dec[:, pred_len:, :], y_pred)
    return loss


# main loop
epoch = 0
while epoch < MAX_EPOCHS:
    # take a batch
    for batch in train_ds:
        num_covs, cat_covs, time_enc, time_dec, target_dec = batch
        
        # zero for target 
        token_dec = target_dec[:, :-pred_len, :]
        zeros = tf.zeros_like(target_dec[:, -pred_len:, :])
        token_target_dec = tf.concat([token_dec, zeros], axis=1)
        
        # feed model
        x_enc = [num_covs, cat_covs, time_enc]
        x_dec = [time_dec, token_target_dec]
        
        # train step
        loss = train_step((x_enc, x_dec), target_dec[:, pred_len:, :])
        
        
    # print loss every epoch
    print(f"Epoch {epoch+1}/{MAX_EPOCHS} training loss: {loss:.4f}, MAE: {train_metrics[0].result():.4f}")
    
    # reset train metrics
    for metric in train_metrics:
        metric.reset_states()
    
    # run validation loop
    for val_batch in val_ds:
        num_covs, cat_covs, time_enc, time_dec, target_dec = val_batch
        
        # zero for target 
        token_dec = target_dec[:, :-pred_len, :]
        zeros = tf.zeros_like(target_dec[:, -pred_len:, :])
        token_target_dec = tf.concat([token_dec, zeros], axis=1)
        
        # feed model
        x_enc = [num_covs, cat_covs, time_enc]
        x_dec = [time_dec, token_target_dec]
        
        # calculate loss
        loss_val = val_step((x_enc, x_dec), target_dec[:, pred_len:, :])
    
        # print loss every epoch
    print(f"Epoch {epoch+1}/{MAX_EPOCHS} validation loss: {loss:.4f}, MAE: {val_metrics[0].result():.4f}")
    
    # reset val metrics
    for metric in val_metrics:
        metric.reset_states()
        
    # update epoch
    epoch += 1
            
