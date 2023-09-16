from model import Preprocessor
import numpy as np

processor = Preprocessor(
    hist_len=720,
    pred_len=96,
    hidden_dims_time_encoder=64,
    output_dims_time_encoder=4,
    local_invariant_vocab_sizes=[7],
    local_invariant_emb_sizes=[4],
    global_vocab_sizes=[],
    global_emb_sizes=[],
    local_variant_vocab_sizes=[],
    local_variant_emb_sizes=[],
    layer_norm=True, 
    dropout_rate=0.
)

B = 32
ts_hist = np.random.rand(B, 720)
num_cov_local_invariant = None
cat_cov_local_invariant = np.random.randint(0, 7, size=(B, 1))
time_features = np.random.rand(720+96, 6)
num_cov_global = None 
cat_cov_global = None
num_cov_local_variant = None 
cat_cov_local_variant = None 

inputs = (ts_hist, num_cov_local_invariant, cat_cov_local_invariant, time_features, num_cov_global, cat_cov_global, num_cov_local_variant, cat_cov_local_variant)
ts_out, local_invariant, global_features, local_variant = processor(inputs)

print(ts_out.shape)
print(local_invariant.shape)
print(global_features.shape)
print(local_variant.shape)