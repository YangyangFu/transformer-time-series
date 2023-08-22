import tensorflow as tf 
import numpy as np

print("numpy ---------------------")

tensor = tf.random.uniform((1, 3, 4, 2))
updates = tf.zeros((1, 3, 2, 2))
# initialize a tensor for indexing
indices = tf.random.uniform((1, 3, 2), maxval=4, dtype=tf.int32)
#indices = 

# check with numpy
original_np = tensor.numpy()
updates_np = updates.numpy()
indices_np = indices.numpy()
original_np[:, 
            np.arange(3).reshape(-1,1), 
            indices_np,
            :] = updates_np

print(original_np)
print(indices_np)
print()

print("===================================")
index_tensor = indices 
update_tensor = updates
original_tensor = tensor


print("tensorflow using tf.Variable()---------------------")
# generate index grid
tensor = tf.Variable(tensor)
for i in range(3):
    update = tf.expand_dims(updates[:, i, :, :], axis=1)
    update_index = tf.expand_dims(indices[:, i, :], axis=1)
    print(update_index)
    print(update)
    for j in range(update_index.shape[-1]):
        print(update_index[0,0,j])
        tensor[:, i, update_index[0,0,j], :].assign(update[0,0,j,:])

print(tensor)
# update the original tensor at given index with update tensor
#updated_tensor = tf.tensor_scatter_nd_update(tensor, indices, updates)

print("tensorflow using tf.tensor_scatter_nd_update()---------------------")
print()

context = tensor
top_index = indices
attention = updates

B, H = context.shape[:2]

# Create the indices for scatter update
b_indices = tf.range(B)[:, tf.newaxis, tf.newaxis, tf.newaxis]
h_indices = tf.range(H)[tf.newaxis, :, tf.newaxis, tf.newaxis]

# Broadcast top_index to match the dimensions
top_expanded = top_index[:, :, :, tf.newaxis]

# Combine indices to form the index tensor
indices = tf.concat([b_indices + tf.zeros_like(top_expanded),
                    h_indices + tf.zeros_like(top_expanded),
                    top_expanded], axis=-1)

# Scatter update
updated_context = tf.tensor_scatter_nd_update(context, indices, updates)

print(updated_context)