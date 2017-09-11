# coding=utf-8
# build a softmax regression model
import tensorflow as tf
import Utils.DataSource as ds

mnist = ds.readMnist("data/")

# using InteractiveSession for interleave operations when building a computation graph
sess = tf.InteractiveSession()

# placeholder
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Session should initialize the variable for using it
sess.run(tf.global_variables_initializer())

# predicted class and loss function
y = tf.matmul(x, W) + b
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = y_))

#  use steepest gradient descent, with a step length of 0.5, to descend the cross entropy.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# using feed_dict to replace the placeholder tensors x and y_
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# evaluate the model
# tf.argmax is an extremely useful function
# which gives you the index of the highest entry in a tensor along some axis
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

