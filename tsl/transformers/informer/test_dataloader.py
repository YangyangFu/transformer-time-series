from data_loader import DataLoader
import tensorflow as tf 

source_seq_len = 64
target_seq_len = 128
pred_len = 96

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

#for i, train_batch in enumerate(train_ds):
#    print(f"train dataset batch {i}, with a size of {train_batch[0].shape}")
max_epochs = 100
for epoch in range(max_epochs):
    for i, val_batch in enumerate(val_ds):
        print(f"val dataset batch {i}, with a size of {val_batch[0].shape}")
        num_covs, cat_covs, time_enc, time_dec, target_dec = val_batch
        # zero for target 
        token_dec = target_dec[:, :-pred_len, :]
        zeros = tf.zeros_like(target_dec[:, -pred_len:, :])
        token_target_dec = tf.concat([token_dec, zeros], axis=1)
    
#for i, test_batch in enumerate(test_ds):
#    print(f"test dataset batch {i}, with a size of {test_batch[0].shape}")

