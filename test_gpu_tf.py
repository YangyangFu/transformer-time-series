import tensorflow as tf

tf.debugging.set_log_device_placement(True)

# Place tensors on the GPU
#with tf.device('/GPU:0'):
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

# Run on the GPU
print(a.shape, b.shape)
c = tf.matmul(a, b)
print(c)
