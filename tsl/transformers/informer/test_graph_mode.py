import tensorflow as tf

class Test(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.linear1 = tf.keras.layers.Dense(5)
        self.linear2 = tf.keras.layers.Dense(5)
        self.add = tf.keras.layers.Add()
        
    def call(self, x):
        x1, x2 = x
        # how to get the shape of x1 for graph mode?
        # cannot analyze the shape of x1 in eager mode inside call() during graph mode
        #b, _= tf.shape(x1)
        b = tf.shape(x1)[0]
        x1 = self.linear1(x1)
        x2 = self.linear2(x2)
        x = self.add([x1, x2])
        
        return x 


x1 = tf.constant([[1,2,3], [4,5,6]])
x2 = tf.constant([[1,2,3], [4,5,6]])
y = tf.random.uniform((2, 5))

model = Test()
y_pred = model([x1, x2])

loss = tf.keras.losses.MeanSquaredError()(y, y_pred)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

epoch = 0

@tf.function
def train_step(x, y):
    x1, x2 = x
    with tf.GradientTape() as tape:
        y_pred = model([x1, x2])
        loss = tf.keras.losses.MeanSquaredError()(y, y_pred)
        
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

while epoch < 10:
    loss = train_step([x1, x2], y)
    
    epoch += 1
    print(loss)
