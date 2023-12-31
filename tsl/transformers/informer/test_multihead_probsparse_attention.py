import tensorflow as tf
import numpy as np
import random
import os 

from multihead_probsparse_attention import MultiHeadProbSparseAttention

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

# create zero padding 
source_sequence = tf.constant([[1, 2, 3, 0, 0], [1, 2, 3, 4, 0]])
target_sequence = tf.constant([[1, 2, 3, 4, 0], [1, 2, 3, 4, 5]])

# create embedding layer
mask_zero = False
emb = tf.keras.layers.Embedding(input_dim=10, output_dim=12, mask_zero=mask_zero)

# create multihead attention layer
mha = MultiHeadProbSparseAttention(num_heads=6,
                            factor=2,
                            key_dim=2,
                            value_dim=2,
                            output_dim=12)

# (2, 3, 4)
query = emb(target_sequence)
# (2, 5, 4)
key = value = emb(source_sequence)

# (2, 3, 4), (2, 2, 3, 5)
out, attention_score  = mha([query, key, value], use_causal_mask=True, return_attention_scores=True)
print(out.shape, attention_score.shape)
print(attention_score)
# check if padding mask is propagated
if mask_zero:
    print(out._keras_mask)
