from dataloader import DataLoader
from model import Informer
import tensorflow as tf

embed_dim = 512
source_seq_len = 64
target_seq_len = 128
pred_len = 96
n_num_covs = 7
n_targets = 1

dataloader = DataLoader(data_path='ETTh1.csv',
                    target_cols=['OT'],
                    num_cov_cols=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'],
                    train_range=(0, 10000),
                    hist_len=source_seq_len,
                    pred_len=pred_len,
                    batch_size=32,
                    )
train_ds = dataloader.generate_dataset(mode="train", shuffle=True, seed=1)

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

# take a batch
for batch in train_ds:
    num_covs, cat_covs, time_enc, time_dec, target_dec = batch
    
    # zero for target 
    token_dec = target_dec[:, :-pred_len, :]
    zeros = tf.zeros_like(target_dec[:, -pred_len:, :])
    target_dec = tf.concat([token_dec, zeros], axis=1)
    
    # feed model
    x_enc = [num_covs, cat_covs, time_enc]
    x_dec = [time_dec, target_dec]
    y = model(x_enc, x_dec)
    print(y.shape)
    print(model.summary())
    
    break
