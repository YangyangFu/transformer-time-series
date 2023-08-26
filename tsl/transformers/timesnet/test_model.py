import tensorflow as tf
from model import TimesNet

x_num = tf.random.normal((2, 10, 7))
x_time = tf.random.normal((2, 10, 7))

model = TimesNet(target_cols_index=[7],
                pred_len=5,
                hist_len=6,
                num_layers=2,
                embedding_dim=16, 
                topk = 2,
                cov_hidden_dim = 32,
                num_kernels = 2,
                num_cat_cov=0, 
                cat_cov_embedding_size=[], 
                cat_cov_embedding_dim=4, 
                use_holiday=False, 
                freq='H', 
                dropout_rate=0.1, 
)

out = model([x_num, None, x_time])
print(out.shape)
