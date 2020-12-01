import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
from scripts.model import ConvNet
from scripts.utils import generate_self_vector

# MNIST dataset parameters.
num_classes = 10  # total classes (0-9 digits).

# Training parameters.
learning_rate = 0.0001
training_steps = 20000
batch_size = 8
display_step = 10

# Network parameters.
conv1_filters = 32  # number of filters for 1st conv layer.
conv2_filters = 64  # number of filters for 2nd conv layer.
fc1_units = 1024  # number of neurons for 1st fully-connected layer.

# Prepare MNIST data.
from tensorflow.keras.datasets import mnist

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert to float32.
#x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, y_train, x_test, y_test = np.load("data/bev5.npy"), np.load("data/pc_with_object_ids5.npy"), \
                                   np.load("data/bev4.npy"), np.load("data/pc_with_object_ids4.npy")
# y_train = y_train[:, :, -1]
# y_temp = np.logical_not(np.logical_and(y_train, np.ones(y_train.shape)))
# y_train = np.array(y_temp, dtype='int')
y_train = generate_self_vector(y_train)
y_test = generate_self_vector(y_test)

# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.
# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

# Build neural network model.
conv_net = ConvNet()

# Cross-Entropy Loss.
# Note that this will apply 'softmax' to the logits.
def cross_entropy_loss(x, y):
    # Convert labels to int 64 for tf cross-entropy function.
    y = tf.cast(y, tf.int64)
    # Apply softmax to logits and compute cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    # Average loss across the batch.
    return tf.reduce_mean(loss)


def mean_square(y_pred, y):
    return tf.reduce_mean(tf.pow(y - y_pred, 2))


# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.Adam(learning_rate)


# Optimization process.
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        # Forward pass.
        pred = conv_net(x, is_training=True)
        # Compute loss.
        loss = tf.reduce_mean(tf.losses.mae(y, pred))
        #loss = cross_entropy_loss()
    # Variables to update, i.e. trainable variables.
    trainable_variables = conv_net.trainable_variables

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))


# Run training for the given number of steps.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # Run the optimization to update W and b values.
    run_optimization(batch_x, batch_y)

    if step % display_step == 0:
        pred = conv_net(batch_x)
        #loss = cross_entropy_loss(pred, batch_y)
        loss = tf.reduce_mean(tf.losses.mse(batch_y, pred))
        #acc = accuracy(pred, batch_y)
        print(step, loss.numpy())

