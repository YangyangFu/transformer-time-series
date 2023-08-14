import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


# how to do this in tensorflow?
import numpy as np
a = np.arange(5*4*3).reshape(5, 4, 3)
indx = [[0, 1],
        [1, 0],
        [2, 0],
        [1, 0]]
g = a[:, [[0],[1],[2],[3]], indx]
print(a)
print(g)
print(g.shape)


a = tf.range(5*4*3)
a = tf.reshape(a, [5, 4, 3])
indx = [[0, 1],
        [1, 0],
        [2, 0],
        [1, 0]]
g1 = a[:, [[0],[1],[2],[3]], indx]
print(g1)
print(g1.shape)
print(ssss)

# Example dimensions
B, T, S, H, D, k = 2, 3, 4, 2, 2, 2

# Example matrix of shape (B, T, S, H, D)
matrix = tf.random.normal((B, T, S, H, D))

# Example index of shape (T, k)
index = tf.random.uniform((T, k), minval=0, maxval=S, dtype=tf.int32)

# Expand dimensions of index to make it broadcastable
index_expanded = tf.reshape(index, (1, T, k, 1, 1))

# Meshgrid for indices
b_indices, t_indices, _, h_indices, d_indices = tf.meshgrid(
    tf.range(B), tf.range(T), tf.range(k), tf.range(H), tf.range(D), indexing='ij'
)

# Form the final indices for tf.gather_nd
final_indices = tf.stack([b_indices, t_indices, index_expanded * tf.ones_like(b_indices), h_indices, d_indices], axis=-1)

# Gather the samples
sampled_matrix = tf.gather_nd(matrix, final_indices)

print("matrix shape:", matrix.shape)
print("sampled_matrix shape:", sampled_matrix.shape)



