import tensorflow as tf
from tensorflow import  keras

# Dense layer from scratch
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__init__()

        # Initialise weights and bias
        self.W = self.add_weight([input_dim, output_dim])
        self.b = self.add_weight([1, output_dim])

    def call(self, inputs):
        #Forward propagate the inputs
        z = tf.matmul(inputs, self.W) + self.B

        #Feed through a non-linear activation
        output = tf.math.sigmoid(z)

        return output


# Tensorflow already has these functions created
layer = tf.keras.layers.Dense(units=2)

# Multi Output Perceptron
model = tf.keras.Sequential([
    tf.keras.layers.Dense(n),
    tf.keras.layers.Dense(2)
])


# Binary Cross Entropy Loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, predicted))

# Mean Squared Error Loss
loss = tf.reduce_mean(tf.square(tf.subtract(y, predicted)))


# Gradient descent
weights = tf.variable((tf.random.normal()))

while True:     # Loop forever
    with tf.GradientTape() as g:
        loss = compute_loss(weights)
        gradient = g.gradient(loss, weights)

    weights = weights - lr * gradient


# Gradient Descent Algorithms
tf.keras.optimizers.SGD # SGD
tf.keras.optimizers.Adam # Adam
tf.keras.optimizers.Adadelta #Adadelta
tf.keras.optimizers.Adagrad #Adagrad
tf.keras.optimizers.RMSprop #RMSProp


# Putting it all together
model = tf.keras.Sequential([...])

# Optimiser
optimizer = tf.keras.optimizers.Adam()

while True:

    # Forward pass through the network
    prediction = model(x)

    with tf.GradientTape() as tape:
        # Compute loss
        loss = compute_loss(y, prediction)

    # update the weights using the gradient
    grads = tape.gradient(loss, model.trainable_varaibles)
    optimizer.appy_gradients(zip(grads, model.trainable_varaibles))


# Regularization
tf.keras.layers.Dropout(p=0.5)

