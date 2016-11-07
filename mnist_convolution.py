from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


# Download mnist data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# Build weight with randomly initialized values
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# Build bias with randomly initialized values
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Calculate convolution with x and w
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# ------------------------------------------------------------------
# Data is feeded into these place holders
x = tf.placeholder(tf.float32, shape=[None, 784])
d = tf.placeholder(tf.float32, shape=[None, 10])
# Toggle dropout
keep_prob = tf.placeholder(tf.float32)

# Define neural network model
# Convolution 1
x_image = tf.reshape(x, [-1, 28, 28, 1])
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# Convolution 2
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# Densely connected layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
# Dropout
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# Readout layer
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
# ------------------------------------------------------------------


def learn():
    # Define session
    session = tf.InteractiveSession()

    # Define how to study (what to optimize)
    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(d * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # Accuracy for console log
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(d, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    session.run(tf.initialize_all_variables())

    # Run learnig
    for i in range(1000):
        batch = mnist.train.next_batch(50)
        # Console log current accuracy
        if i % 50 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], d: batch[1], keep_prob: 1.0})
            print("step %d: training accuracy %g" % (i, train_accuracy))

        train_step.run(feed_dict={x: batch[0], d: batch[1], keep_prob: 1.0})

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, d: mnist.test.labels, keep_prob: 0.5}))


def main():
    learn()

if __name__ == '__main__':
    main()
