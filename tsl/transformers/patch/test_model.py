import tensorflow as tf 

from model import PatchTST

pred_len = 96
hist_len = 512
num_features = 8
target_col_index = [0,1,2,6,7]
num_targets = len(target_col_index) 
embedding_dim = 16
num_layers = 3
num_heads = 4
ffn_hidden_dim = 128
patch_size = 16 
patch_strides = 8 
patch_padding = "end" 
dropout_rate = 0.2

batch_size = 32

# generate fake data
x = tf.random.normal((batch_size, hist_len, num_features))
y = tf.random.normal((batch_size, pred_len, num_targets))

# create model
model = PatchTST(
    pred_len=pred_len,
    target_col_index=target_col_index,
    embedding_dim=embedding_dim,
    num_layers=num_layers,
    num_heads=num_heads,
    ffn_hidden_dim=ffn_hidden_dim,
    patch_size=patch_size,
    patch_strides=patch_strides,
    patch_padding=patch_padding,
    dropout_rate=dropout_rate
    )

y_pred = model(x)
print(y_pred.shape)
    
